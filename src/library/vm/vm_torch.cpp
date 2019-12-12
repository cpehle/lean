/*
 Copyright (c) 2017 Christian Pehle. All rights reserved.
 Released under Apache 2.0 license as described in the file LICENSE.
 Author: Christian Pehle
 */
#include "library/vm/vm.h"
#include "library/vm/vm_nat.h"
#include "library/vm/vm_list.h"
#include "library/vm/vm_torch.h"
#include "library/vm/vm_string.h"

namespace lean {

struct vm_torch : public vm_external {
    torch::Tensor m_val;
    vm_torch(torch::Tensor const & v): m_val(v) {}
    virtual ~vm_torch() {}
    virtual void dealloc() override { this->~vm_torch(); get_vm_allocator().deallocate(sizeof(vm_torch), this); }
    virtual vm_external * ts_clone(vm_clone_fn const &) override { return new vm_torch(m_val); }
    virtual vm_external * clone(vm_clone_fn const &) override { return new (get_vm_allocator().allocate(sizeof(vm_torch))) vm_torch(m_val); } 
};

vm_obj to_obj(torch::Tensor const & v) {
    return mk_vm_external(new (get_vm_allocator().allocate(sizeof(vm_torch))) vm_torch(v));
}

torch::Tensor const & to_torch(vm_obj const & o) {
    lean_assert(is_external(o));
    lean_assert(dynamic_cast<vm_torch*>(to_external(o)));
    return static_cast<vm_torch*>(to_external(o))->m_val;
}

vm_obj torch_dummy() {
     throw exception("torch_dummy not supposed to be called");
     return mk_vm_unit();
 }

// is_tensor
// numel


vm_obj torch_to_repr(vm_obj const & shape, vm_obj const & v) {
    std::ostringstream out;
    out << to_torch(v);
    return to_obj(out.str());
}

// creation ops
// zeros
vm_obj torch_zeros(vm_obj const & shape) {
    list<unsigned> dims = to_list<unsigned, std::function<unsigned(vm_obj const &)> >(shape, to_unsigned);
    std::vector<long long> d;
    for (auto dim : dims) {
        d.push_back(dim);
    }

    return to_obj(torch::zeros(torch::IntArrayRef(d)));
}
// zeros_like
// ones
vm_obj torch_ones(vm_obj const & shape) {
    list<unsigned> dims = to_list<unsigned, std::function<unsigned(vm_obj const &)> >(shape, to_unsigned);
    std::vector<long long> d;
    for (auto dim : dims) {
        d.push_back(dim);
    }

    return to_obj(torch::ones(torch::IntArrayRef(d)));
}
// ones_like

// random generation
// randn
vm_obj torch_randn(vm_obj const & shape) {
    list<unsigned> dims = to_list<unsigned, std::function<unsigned(vm_obj const &)> >(shape, to_unsigned);
    std::vector<long long> d;
    for (auto dim : dims) {
        d.push_back(dim);
    }

    return to_obj(torch::randn(torch::IntArrayRef(d)));
}


// indexing, slicing, joining, mutating Ops

// cat
// chunk
// gather
// ... 


// math operations

// abs
vm_obj torch_abs(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::abs(to_torch(x))); }
// acos
vm_obj torch_acos(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::acos(to_torch(x))); }
// add
vm_obj torch_add(vm_obj const & /* shape */, vm_obj const & x, vm_obj const & y) { return to_obj(to_torch(x) + to_torch(y)); }
// addcdiv
// addcmul
// asin
// atan
// atan2
// bitwise_not
// ceil
// clamp
// cos

vm_obj torch_grad(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(to_torch(x).grad()); }


vm_obj torch_mul(vm_obj const & /* shape */, vm_obj const & x, vm_obj const & y) { return to_obj(to_torch(x) * to_torch(y)); }

vm_obj torch_transpose(vm_obj const & /* m */, vm_obj const & /* n */, vm_obj const & x) { return to_obj(torch::t(to_torch(x))); }
vm_obj torch_mm(vm_obj const & /* m */, vm_obj const & /* n */, vm_obj const & /* p */, vm_obj const & x, vm_obj const & y) { return to_obj(torch::mm(to_torch(x), to_torch(y))); }
vm_obj torch_bmm(vm_obj const & /* b */, vm_obj const & /* m */, vm_obj const & /* n */, vm_obj const & /* p */, vm_obj const & x, vm_obj const & y) { return to_obj(torch::bmm(to_torch(x), to_torch(y))); }



void initialize_vm_torch() {
    DECLARE_VM_BUILTIN(name({"torch", "T"}), torch_dummy);
    DECLARE_VM_BUILTIN(name({"torch", "T", "repr"}), torch_to_repr);
    DECLARE_VM_BUILTIN(name({"torch", "T", "zeros"}), torch_zeros);
    DECLARE_VM_BUILTIN(name({"torch", "T", "ones"}), torch_ones);

    // random
    DECLARE_VM_BUILTIN(name({"torch", "T", "randn"}), torch_randn);

    // mathematical operations
    DECLARE_VM_BUILTIN(name({"torch", "T", "abs"}), torch_abs);
    DECLARE_VM_BUILTIN(name({"torch", "T", "add"}), torch_add);
    DECLARE_VM_BUILTIN(name({"torch", "T", "mul"}), torch_mul);

    DECLARE_VM_BUILTIN(name({"torch", "T", "mm"}), torch_mm);
    DECLARE_VM_BUILTIN(name({"torch", "T", "bmm"}), torch_bmm);


    // there is two: torch.t torch.transpose
    DECLARE_VM_BUILTIN(name({"torch", "T", "transpose"}), torch_transpose);
}
void finalize_vm_torch() {
}
}