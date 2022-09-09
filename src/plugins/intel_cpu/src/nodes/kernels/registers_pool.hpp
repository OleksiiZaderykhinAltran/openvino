// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"
#include <dnnl_types.h>
#include "ie_common.h"
#include <utility>

namespace ov {
namespace intel_cpu {

using namespace dnnl::impl::cpu;

/**
 * The RegistersPool is the base class for the IsaRegistersPool template:
 *      template <x64::cpu_isa_t isa>
 *      class IsaRegistersPool : public RegistersPool;
 *
 * The registers pool must be created by instantiating the IsaRegistersPool template, like the next:
 *      RegistersPool::Ptr regPool = RegistersPool::create<isa>({
 *          // the list of the registers to be excluded from pool
 *          Reg64(Operand::RAX), Reg64(Operand::RCX), Reg64(Operand::RDX), Reg64(Operand::RBX),
 *          Reg64(Operand::RSP), Reg64(Operand::RBP), Reg64(Operand::RSI), Reg64(Operand::RDI)
 *      });
 */
class RegistersPool {
public:
    using Ptr = std::shared_ptr<RegistersPool>;
    static constexpr int anyIdx = -1;

    /**
     * The scoped wrapper for the Xbyak registers.
     * By creating it you are getting the register from the pool RegistersPool.
     * It could be created by using constructor with RegistersPool as an argument, like the next:
     *      const RegistersPool::Reg<Xbyak::Reg64> reg {regPool};
     * The destructor will return the register to the pool. Or it could be returned manually:
     *      reg.release();
     * @tparam TReg Xbyak register class
     */
    template<typename TReg>
    class Reg {
        friend class RegistersPool;
    public:
        Reg() {
            static_assert(std::is_same<TReg, Xbyak::Xmm>::value || std::is_same<TReg, Xbyak::Ymm>::value || std::is_same<TReg, Xbyak::Zmm>::value ||
                          std::is_same<TReg, Xbyak::Reg64>::value || std::is_same<TReg, Xbyak::Reg32>::value || std::is_same<TReg, Xbyak::Opmask>::value,
                          "The type is not supported for the RegistersPool::Reg template");
        }
        Reg(const RegistersPool::Ptr& regPool) { initialize(regPool); }
        Reg(const RegistersPool::Ptr& regPool, int requestedIdx) { initialize(regPool, requestedIdx); }
        ~Reg() { release(); }
        Reg& operator=(Reg&& other)  noexcept {
            reg = other.reg;
            regPool = std::move(other.regPool);
            return *this;
        }
        Reg(Reg&& other)  noexcept : reg(other.reg), regPool(std::move(other.regPool)) {}
        operator TReg&() { ensureValid(); return reg; }
        operator const TReg&() const { ensureValid(); return reg; }
        operator Xbyak::RegExp() const { ensureValid(); return reg; }
        int getIdx() const { ensureValid(); return reg.getIdx(); }
        friend Xbyak::RegExp operator+(const Reg& lhs, const Xbyak::RegExp& rhs) {
            lhs.ensureValid();
            return lhs.operator Xbyak::RegExp() + rhs;
        }
        void release() {
            if (regPool) {
                regPool->returnToPool(reg);
                regPool.reset();
            }
        }

    private:
        void ensureValid() const {
            if (!regPool) {
                IE_THROW() << "RegistersPool::Reg is either not initialized or released";
            }
        }

        void initialize(const RegistersPool::Ptr& pool, int requestedIdx = anyIdx) {
            release();
            reg = TReg(pool->template getFree<TReg>(requestedIdx));
            regPool = pool;
        }

    private:
        TReg reg;
        RegistersPool::Ptr regPool;
    };

    virtual ~RegistersPool() {
        checkUniqueAndUpdate(false);
    }

    template <x64::cpu_isa_t isa>
    static Ptr create(std::initializer_list<Xbyak::Reg> regsToExclude);

    template<typename TReg>
    size_t countFree() const {
        if (std::is_same<TReg, Xbyak::Xmm>::value || std::is_same<TReg, Xbyak::Ymm>::value || std::is_same<TReg, Xbyak::Zmm>::value) {
            return simdSet.countUnused();
        } else if (std::is_same<TReg, Xbyak::Reg64>::value || std::is_same<TReg, Xbyak::Reg32>::value) {
            return generalSet.countUnused();
        } else if (std::is_same<TReg, Xbyak::Opmask>::value) {
            return countUnusedOpmask();
        }
    }

protected:
    class PhysicalSet {
    public:
        PhysicalSet(int size) : isFreeIndexVector(size, true) {}

        void setAsUsed(int regIdx) {
            if (regIdx >= isFreeIndexVector.size() || regIdx < 0) {
                IE_THROW() << "regIdx is out of bounds in RegistersPool::PhysicalSet::setAsUsed()";
            }
            if (!isFreeIndexVector[regIdx]) {
                IE_THROW() << "Inconsistency in RegistersPool::PhysicalSet::setAsUsed()";
            }
            isFreeIndexVector[regIdx] = false;
        }

        void setAsUnused(int regIdx) {
            if (regIdx >= isFreeIndexVector.size() || regIdx < 0) {
                IE_THROW() << "regIdx is out of bounds in RegistersPool::PhysicalSet::setAsUsed()";
            }
            if (isFreeIndexVector[regIdx]) {
                IE_THROW() << "Inconsistency in RegistersPool::PhysicalSet::setAsUnused()";
            }
            isFreeIndexVector[regIdx] = true;
        }

        int getUnused(int requestedIdx) {
            if (requestedIdx == anyIdx) {
                return getFirstFreeIndex();
            } else {
                if (!isFreeIndexVector[requestedIdx]) {
                    IE_THROW() << "The register with index #" << requestedIdx << " already used in the RegistersPool";
                }
                return requestedIdx;
            }
        }

        void exclude(Xbyak::Reg reg) {
            isFreeIndexVector.at(reg.getIdx()) = false;
        }

        size_t countUnused() const {
            size_t count = 0;
            for (const auto& isFree : isFreeIndexVector) {
                if (isFree) {
                    ++count;
                }
            }
            return count;
        }

    private:
        int getFirstFreeIndex() {
            for (int c = 0; c < isFreeIndexVector.size(); ++c) {
                if (isFreeIndexVector[c]) {
                    return c;
                }
            }
            IE_THROW() << "Not enough registers in the RegistersPool";
        }

    private:
        std::vector<bool> isFreeIndexVector;
    };

    virtual int getFreeOpmask(int requestedIdx) { IE_THROW() << "getFreeOpmask: The Opmask is not supported in current instruction set"; }
    virtual void returnOpmaskToPool(int idx) { IE_THROW() << "returnOpmaskToPool: The Opmask is not supported in current instruction set"; }
    virtual size_t countUnusedOpmask() const { IE_THROW() << "countUnusedOpmask: The Opmask is not supported in current instruction set"; }

    RegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude, int simdRegistersNumber)
            : simdSet(simdRegistersNumber) {
        checkUniqueAndUpdate();
        for (auto& reg : regsToExclude) {
            if (reg.isXMM() || reg.isYMM() || reg.isZMM()) {
                simdSet.exclude(reg);
            } else if (reg.isREG()) {
                generalSet.exclude(reg);
            }
        }
        generalSet.exclude(Xbyak::Reg64(Xbyak::Operand::RSP));
    }

private:
    template<typename TReg>
    int getFree(int requestedIdx) {
        static_assert(std::is_same<TReg, Xbyak::Xmm>::value || std::is_same<TReg, Xbyak::Ymm>::value || std::is_same<TReg, Xbyak::Zmm>::value ||
                      std::is_same<TReg, Xbyak::Reg64>::value || std::is_same<TReg, Xbyak::Reg32>::value || std::is_same<TReg, Xbyak::Opmask>::value,
                      "Unsupported TReg by RegistersPool. Please, use the following registers either Reg32, Reg64, Xmm, Ymm, Zmm or Opmask");
        if (std::is_same<TReg, Xbyak::Xmm>::value || std::is_same<TReg, Xbyak::Ymm>::value || std::is_same<TReg, Xbyak::Zmm>::value) {
            auto idx = simdSet.getUnused(requestedIdx);
            simdSet.setAsUsed(idx);
            return idx;
        } else if (std::is_same<TReg, Xbyak::Reg64>::value || std::is_same<TReg, Xbyak::Reg32>::value) {
            auto idx = generalSet.getUnused(requestedIdx);
            generalSet.setAsUsed(idx);
            return idx;
        } else if (std::is_same<TReg, Xbyak::Opmask>::value) {
            return getFreeOpmask(requestedIdx);
        }
    }

    template<typename TReg>
    void returnToPool(TReg reg) {
        if (std::is_same<TReg, Xbyak::Xmm>::value || std::is_same<TReg, Xbyak::Ymm>::value || std::is_same<TReg, Xbyak::Zmm>::value) {
            simdSet.setAsUnused(reg.getIdx());
        } else if (std::is_same<TReg, Xbyak::Reg64>::value || std::is_same<TReg, Xbyak::Reg32>::value) {
            generalSet.setAsUnused(reg.getIdx());
        } else if (std::is_same<TReg, Xbyak::Opmask>::value) {
            returnOpmaskToPool(reg.getIdx());
        }
    }

    void checkUniqueAndUpdate(bool isCtor = true) {
        static thread_local bool isCreated = false;
        if (isCtor) {
            if (isCreated) {
                IE_THROW() << "There should be only one instance of RegistersPool per thread";
            }
            isCreated = true;
        } else {
            isCreated = false;
        }
    }

    PhysicalSet generalSet {16};
    PhysicalSet simdSet;
};

template <x64::cpu_isa_t isa>
class IsaRegistersPool : public RegistersPool {
public:
    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude) : RegistersPool(regsToExclude, x64::cpu_isa_traits<isa>::n_vregs) {}
};

template <>
class IsaRegistersPool<x64::avx512_core> : public RegistersPool {
public:
    IsaRegistersPool(std::initializer_list<Xbyak::Reg> regsToExclude)
            : RegistersPool(regsToExclude, x64::cpu_isa_traits<x64::avx512_core>::n_vregs) {
        for (auto& reg : regsToExclude) {
            if (reg.isOPMASK()) {
                opmaskSet.exclude(reg);
            }
        }
    }

    int getFreeOpmask(int requestedIdx) override {
        auto idx = opmaskSet.getUnused(requestedIdx);
        opmaskSet.setAsUsed(idx);
        return idx;
    }

    void returnOpmaskToPool(int idx) override {
        opmaskSet.setAsUnused(idx);
    }

    size_t countUnusedOpmask() const override {
        return opmaskSet.countUnused();
    }

protected:
    PhysicalSet opmaskSet {8};
};

template <x64::cpu_isa_t isa>
RegistersPool::Ptr RegistersPool::create(std::initializer_list<Xbyak::Reg> regsToExclude) {
    return std::make_shared<IsaRegistersPool<isa>>(regsToExclude);
}

}   // namespace intel_cpu
}   // namespace ov
