// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_depth_to_space_node.h"

#include <mkldnn_extension_utils.h>
#include <utils/general_utils.h>

#include <cmath>
#include <common/primitive_hashing_utils.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <string>

#include "common/blocked_desc_creator.h"

#define THROW_ERROR IE_THROW() << "DepthToSpace layer with name '" << getName() << "' "

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;

size_t MKLDNNDepthToSpaceNode::DepthToSpaceAttrs::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, layoutType);
    seed = hash_combine(seed, mode);
    seed = hash_combine(seed, blockSize);
    seed = hash_combine(seed, blockStep);
    seed = hash_combine(seed, dataSize);
    seed = hash_combine(seed, nSpatialDims);
    seed = get_vector_hash(seed, srcBlockedDims);

    return seed;
}

bool MKLDNNDepthToSpaceNode::DepthToSpaceAttrs::operator==(const DepthToSpaceAttrs& rhs) const {
    bool result = layoutType == rhs.layoutType && mode == rhs.mode &&
                  blockSize == rhs.blockSize && blockStep == rhs.blockStep &&
                  dataSize == rhs.dataSize && nSpatialDims == rhs.nSpatialDims &&
                  srcBlockedDims == rhs.srcBlockedDims;

    return result;
}

bool MKLDNNDepthToSpaceNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto depthToSpace = ov::as_type_ptr<const ngraph::opset1::DepthToSpace>(op);
        if (!depthToSpace) {
            errorMessage = "Only opset1 DepthToSpace operation is supported";
            return false;
        }
        const auto mode = depthToSpace->get_mode();
        if (!one_of(mode, ngraph::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, ngraph::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST)) {
            errorMessage = "Does not support mode: " + ngraph::as_string(mode);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNDepthToSpaceNode::MKLDNNDepthToSpaceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    if (inputShapes.size() != 1 || outputShapes.size() != 1)
        THROW_ERROR << "has incorrect number of input/output edges!";

    auto depthToSpace = ov::as_type_ptr<const ngraph::opset1::DepthToSpace>(op);
    if (!depthToSpace)
        THROW_ERROR << "supports only opset1";

    const auto modeNgraph = depthToSpace->get_mode();
    if (modeNgraph == ngraph::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST) {
        attrs.mode = Mode::BLOCKS_FIRST;
    } else if (modeNgraph == ngraph::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST) {
        attrs.mode = Mode::DEPTH_FIRST;
    } else {
        THROW_ERROR << "doesn't support mode: " << ngraph::as_string(modeNgraph);
    }

    attrs.blockSize = depthToSpace->get_block_size();
    if (attrs.blockSize == 0)
        THROW_ERROR << "has incorrect block_size parameter is zero!";

    const size_t srcRank = getInputShapeAtPort(0).getRank();
    const size_t dstRank = getOutputShapeAtPort(0).getRank();

    if (srcRank < 3)
        THROW_ERROR << "has incorrect number of input dimensions";
    if (srcRank > 5)
        THROW_ERROR << "doesn't support dimensions with rank greater than 5";
    if (srcRank != dstRank)
        THROW_ERROR << "has incorrect number of input/output dimensions";

    const size_t nSpatialDims = srcRank - 2;
    attrs.blockStep = static_cast<size_t>(std::pow(attrs.blockSize, nSpatialDims));
}

void MKLDNNDepthToSpaceNode::getSupportedDescriptors() {}

void MKLDNNDepthToSpaceNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(0);

    impl_desc_type impl_type = impl_desc_type::ref;
    if (cpu::x64::mayiuse(cpu::x64::avx512_common)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (cpu::x64::mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (cpu::x64::mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    }

    NodeConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;

    const auto& inputDataShape = getInputShapeAtPort(0);
    const auto& outputDataShape = getOutputShapeAtPort(0);

    std::vector<LayoutType> supportedTypes;
    if (inputDataShape.getRank() > 2) {
        const auto& srcDims = inputDataShape.getDims();
        auto canUseBlocked = [=](const size_t block) {
            return srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % block == 0 && (srcDims[1] / block) % attrs.blockStep == 0 &&
                   (attrs.mode == Mode::DEPTH_FIRST ? block % attrs.blockStep == 0 : true);
        };

        supportedTypes.push_back(LayoutType::nspc);
        if (canUseBlocked(8lu))
            supportedTypes.push_back(LayoutType::nCsp8c);
        if (canUseBlocked(16lu))
            supportedTypes.push_back(LayoutType::nCsp16c);
    }
    supportedTypes.push_back(LayoutType::ncsp);
    auto creators = BlockedDescCreator::getCommonCreators();
    auto range = BlockedDescCreator::makeFilteredRange(creators, inputDataShape.getRank(), supportedTypes);

    for (auto itr = range.first; itr != range.second; ++itr) {
        config.inConfs[0].desc = itr->second->createSharedDesc(precision, inputDataShape);
        config.outConfs[0].desc = itr->second->createSharedDesc(precision, outputDataShape);
        supportedPrimitiveDescriptors.emplace_back(config, impl_type);
    }
}

void MKLDNNDepthToSpaceNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_ERROR << "has not allocated destination memory";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_ERROR << "has not allocated input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << "has unidentified preferable primitive descriptor";

    const auto& memoryDesc = srcMemPtr->getDesc();
    attrs.dataSize = memoryDesc.getPrecision().size();
    attrs.nSpatialDims = memoryDesc.getShape().getRank() - 2;
    attrs.layoutType = memoryDesc.hasLayoutType(LayoutType::nCsp16c) ? LayoutType::nCsp16c :
                       memoryDesc.hasLayoutType(LayoutType::nCsp8c) ? LayoutType::nCsp8c :
                       memoryDesc.hasLayoutType(LayoutType::nspc) ? LayoutType::nspc : LayoutType::ncsp;

    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

void MKLDNNDepthToSpaceNode::prepareParams() {
    attrs.srcBlockedDims = getParentEdgeAt(0)->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>()->getBlockDims();
    auto builder = [](const DepthToSpaceAttrs& key) -> std::shared_ptr<DepthToSpaceExecutor> {
        return std::make_shared<DepthToSpaceExecutor>(key);
    };

    auto cache = getRuntimeCache();
    auto result = cache->getOrCreate(attrs, builder);
    if (!result.first) {
        IE_THROW() << "DepthToSpaceExecutor was not found for node " << getName() << ".";
    }

    execPtr = result.first;
}

MKLDNNDepthToSpaceNode::DepthToSpaceExecutor::DepthToSpaceExecutor(const DepthToSpaceAttrs& attrs) {
    if (!MKLDNNPlugin::one_of(attrs.layoutType, LayoutType::nCsp16c, LayoutType::nCsp8c, LayoutType::nspc, LayoutType::ncsp))
        IE_THROW() << "DepthToSpace executor supports only 'nCsp16c', 'nCsp8c', 'nspc' or 'ncsp' layouts.";

    const bool isBlocked = MKLDNNPlugin::one_of(attrs.layoutType, LayoutType::nCsp16c, LayoutType::nCsp8c);
    const bool isChannelsFirst = attrs.layoutType == LayoutType::nspc;
    const size_t nDims = attrs.srcBlockedDims.size();
    const size_t reshapedRank = nDims + attrs.nSpatialDims + static_cast<int>(isBlocked && attrs.mode == Mode::DEPTH_FIRST);
    const size_t lastIdx = reshapedRank - 1;
    size_t firstSpatialOrder = 2;

    PermuteParams params;
    params.data_size = attrs.dataSize;
    params.order.resize(reshapedRank, 0);
    params.src_block_order.resize(reshapedRank);
    params.dst_block_order.resize(reshapedRank);
    params.dst_block_dims.resize(reshapedRank);
    params.src_block_dims.resize(reshapedRank);
    params.src_block_dims[0] = attrs.srcBlockedDims[0];

    // reshaping of src dimensions and creating the permutation order for each layout:
    // new shape: mode = blocks_first [N, block_size, block_size, ..., block_size, C / (block_size ^ K), D1, D2, ..., DK]
    //            mode = depth_first  [N, C / (block_size ^ K), block_size, block_size, ..., block_size, D1, D2, ..., DK]
    // order    : mode = blocks_first : [0,  K + 1,  K + 2, 1, K + 3, 2, K + 4, 3, ..., K + (K + 1), K]
    //            mode = depth_first  : [0,  1,  K + 2, 2, K + 3, 3, K + 4, 4, ..., K + (K + 1), K + 1]
    // where `k` is number of spatial dimensions

    auto reshapeAndSetPermOrder = [&](const size_t idx1, const size_t idx2, const size_t shift, const VectorDims& dims) {
        for (size_t i = 0; i < attrs.nSpatialDims; i++) {
            params.order[i * 2 + shift] = i + idx1;
            params.order[i * 2 + shift + 1] = i + idx2;

            params.src_block_dims[params.order[i * 2 + shift]] = dims[i + shift];
            params.src_block_dims[params.order[i * 2 + shift + 1]] = attrs.blockSize;
        }
    };

    if (isBlocked) {
        size_t orderShiftForBlocks, orderShiftForDims;
        if (attrs.mode == Mode::BLOCKS_FIRST) {
            orderShiftForBlocks = 1;
            orderShiftForDims = attrs.nSpatialDims + 2;

            params.src_block_dims[attrs.nSpatialDims + 1] = attrs.srcBlockedDims[1] / attrs.blockStep;
            params.src_block_dims[lastIdx] = attrs.srcBlockedDims.back();

            params.order[1] = attrs.nSpatialDims + 1;
            params.order[lastIdx] = lastIdx;
        } else {
            orderShiftForBlocks = attrs.nSpatialDims + 4;
            orderShiftForDims = 3;

            size_t newBlockSize = attrs.srcBlockedDims.back() / attrs.blockStep;
            size_t newBlocksCount = attrs.srcBlockedDims[1] / attrs.blockStep;
            params.src_block_dims[1] = newBlocksCount;
            params.src_block_dims[2] = attrs.srcBlockedDims[1] / newBlocksCount;
            params.src_block_dims[lastIdx - attrs.nSpatialDims] = newBlockSize;

            params.order[1] = 1;
            params.order[2] = 3;
            params.order[lastIdx - 1] = 2;
            params.order[lastIdx] = lastIdx - attrs.nSpatialDims;
        }

        reshapeAndSetPermOrder(orderShiftForDims, orderShiftForBlocks, firstSpatialOrder, attrs.srcBlockedDims);
    } else if (isChannelsFirst) {
        firstSpatialOrder = 1;

        size_t shift = static_cast<size_t>(attrs.mode == DEPTH_FIRST) + attrs.nSpatialDims + 1;
        params.order[lastIdx] = attrs.mode == Mode::DEPTH_FIRST ? attrs.nSpatialDims + 1 : lastIdx;
        params.src_block_dims[params.order[lastIdx]] = attrs.srcBlockedDims.back() / attrs.blockStep;

        reshapeAndSetPermOrder(firstSpatialOrder, shift, firstSpatialOrder, attrs.srcBlockedDims);
    } else {
        size_t shift = static_cast<size_t>(attrs.mode == DEPTH_FIRST) + 1;
        params.order[1] = attrs.mode == DEPTH_FIRST ? 1 : attrs.nSpatialDims + 1;
        params.src_block_dims[params.order[1]] = attrs.srcBlockedDims[1] / attrs.blockStep;

        reshapeAndSetPermOrder(attrs.nSpatialDims + firstSpatialOrder, shift, firstSpatialOrder, attrs.srcBlockedDims);
    }

    std::iota(params.src_block_order.begin(), params.src_block_order.end(), 0);
    std::iota(params.dst_block_order.begin(), params.dst_block_order.end(), 0);
    for (size_t i = 0; i < reshapedRank; i++)
        params.dst_block_dims[i] = params.src_block_dims[params.order[i]];

    permuteKernel = std::unique_ptr<PermuteKernel>(new PermuteKernel(params));
}

void MKLDNNDepthToSpaceNode::DepthToSpaceExecutor::exec(MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr, const int MB) {
    if (!permuteKernel)
        IE_THROW() << "Could not execute. Kernel for Transpose node was not compiled.";

    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(srcMemPtr->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(dstMemPtr->GetPtr());

    permuteKernel->execute(srcData, dstData, MB);
}

void MKLDNNDepthToSpaceNode::execute(mkldnn::stream strm) {
    if (!execPtr) {
        THROW_ERROR << "doesn't have a compiled executor.";
    }

    int MB = isDynamicNode() ? getParentEdgeAt(0)->getMemoryPtr()->getStaticDims()[0] : batchToProcess();
    execPtr->exec(getParentEdgeAt(0)->getMemoryPtr(), getChildEdgeAt(0)->getMemoryPtr(), MB);
}

void MKLDNNDepthToSpaceNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

bool MKLDNNDepthToSpaceNode::created() const {
    return getType() == DepthToSpace;
}
REG_MKLDNN_PRIM_FOR(MKLDNNDepthToSpaceNode, DepthToSpace);