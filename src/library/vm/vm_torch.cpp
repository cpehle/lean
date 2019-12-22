/*
 Copyright (c) 2017 Christian Pehle. All rights reserved.
 Released under Apache 2.0 license as described in the file LICENSE.
 Author: Christian Pehle
 */
#include "library/vm/vm.h"
#include "library/vm/vm_nat.h"
#include "library/vm/vm_array.h"
#include "library/vm/vm_list.h"
#include "library/vm/vm_torch.h"
#include "library/vm/vm_string.h"

#include "library/parray.h"

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


vm_obj torch_to_repr(vm_obj const & /*shape*/, vm_obj const & v) {
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
vm_obj torch_zeros_like(vm_obj const & /* shape */, vm_obj const & tensor) {
    return to_obj(torch::zeros_like(to_torch(tensor)));
}
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
vm_obj torch_ones_like(vm_obj const & /* shape */, vm_obj const & tensor) {
    return to_obj(torch::ones_like(to_torch(tensor)));
}

// random generation
// randn
vm_obj torch_randn(vm_obj const & shape) {
    list<unsigned> dims = to_list<unsigned, std::function<unsigned(vm_obj const &)> >(shape, to_unsigned);
    std::vector<long long> d;
    for (auto dim : dims) {
        d.push_back(dim);
    }

    return to_obj(torch::randn(torch::IntArrayRef(d), torch::TensorOptions().requires_grad(true)));
}


// indexing, slicing, joining, mutating Ops

// cat
// chunk
// gather
// ... 

// autograd
vm_obj grad(vm_obj const & /* shape */, vm_obj const & /* shape */, vm_obj const & output, vm_obj const & input, vm_obj const & grad_output) {
    torch::autograd::variable_list out_v({to_torch(output)});
    torch::autograd::variable_list in_v({to_torch(input)});
    torch::autograd::variable_list grad_out_v({to_torch(grad_output)});    
    auto grad_in_v = torch::autograd::grad(out_v, in_v, grad_out_v);

    return to_obj(grad_in_v[0]);
}

vm_obj backward(vm_obj const & /* shape */, vm_obj const & output, vm_obj const & grad_output) {
    auto out = to_torch(output);
    out.backward(to_torch(grad_output));
    return output;
}

vm_obj grad_of(vm_obj const & /* shape */, vm_obj const & x) {
    return to_obj(to_torch(x).grad());
}

//
vm_obj torch_stack(vm_obj const & /* shape */, vm_obj const & x) {
    list<torch::Tensor> tensors = to_list<torch::Tensor, std::function<torch::Tensor(vm_obj const &)> >(x, to_torch);
    std::vector<torch::Tensor> tensors_vec;
    for (auto t : tensors) {
        tensors_vec.push_back(t);
    }

    return to_obj(torch::stack(torch::ArrayRef<torch::Tensor>(tensors_vec)));
}



// vm_obj scan(vm_obj const & /*shape*/, vm_obj const & fn, vm_obj const & in, vm_obj const & s, vm_obj const & dim) {
//     auto dim = to_unsigned(dim);
//     auto x = to_torch(in);
//     auto shape = x.sizes();
// 
//     std::vector<torch::Tensor> result;
//     int N = shape[dim];
// 
//     for (int i = 0; i < N; i++) {
//         // fn : (in, s) -> (out, s)
//         auto res = invoke(fn, mk_pair(to_obj(x(dim)), s));
//         auto out = to_torch(cfield(res, 0));
//         s = cfield(res, 1);
//         result.push_back(out);
//     }
// 
//     return mk_vm_pair(to_obj(torch::stack(result)), s);
// }

// math operations
vm_obj torch_abs(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::abs(to_torch(x)));}
vm_obj torch_acos(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::acos(to_torch(x)));}
vm_obj torch_add(vm_obj const & /* shape */, vm_obj const & x, vm_obj const & y) { return to_obj(to_torch(x) + to_torch(y));}
vm_obj torch_asin(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::asin(to_torch(x)));}
vm_obj torch_atan(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::atan(to_torch(x)));}
vm_obj torch_bitwise_not(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::bitwise_not(to_torch(x)));}
vm_obj torch_ceil(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::ceil(to_torch(x)));}
vm_obj torch_cos(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::cos(to_torch(x)));}
vm_obj torch_cosh(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::cosh(to_torch(x)));}
vm_obj torch_div(vm_obj const & /* shape */, vm_obj const & x, vm_obj const & y) { return to_obj(torch::div(to_torch(x), to_torch(y)));}
vm_obj torch_digamma(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::digamma(to_torch(x)));}
vm_obj torch_erf(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::erf(to_torch(x)));}
vm_obj torch_erfc(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::erfc(to_torch(x)));}
vm_obj torch_erfinv(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::erfinv(to_torch(x)));}
vm_obj torch_exp(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::exp(to_torch(x)));}
vm_obj torch_expm1(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::expm1(to_torch(x)));}
vm_obj torch_floor(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::floor(to_torch(x)));}
vm_obj torch_fmod(vm_obj const & /* shape */, vm_obj const & x, vm_obj const & y) { return to_obj(torch::fmod(to_torch(x), to_torch(y)));}
vm_obj torch_frac(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::frac(to_torch(x)));}
// vm_obj torch_lerp(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::lerp(to_torch(x)));}
vm_obj torch_lgamma(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::lgamma(to_torch(x)));}
vm_obj torch_log(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::log(to_torch(x)));}
vm_obj torch_log10(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::log10(to_torch(x)));}
vm_obj torch_log1p(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::log1p(to_torch(x)));}
vm_obj torch_log2(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::log2(to_torch(x)));}
vm_obj torch_logical_not(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::logical_not(to_torch(x)));}
vm_obj torch_logical_xor(vm_obj const & /* shape */, vm_obj const & x, vm_obj const & y) { return to_obj(torch::logical_xor(to_torch(x), to_torch(y)));}
vm_obj torch_mul(vm_obj const & /* shape */, vm_obj const & x, vm_obj const & y) { return to_obj(to_torch(x) * to_torch(y)); }
vm_obj torch_neg(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::neg(to_torch(x)));}
// vm_obj torch_pow(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::pow(to_torch(x)));}
vm_obj torch_reciprocal(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::reciprocal(to_torch(x)));}
// vm_obj torch_remainder(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::remainder(to_torch(x)));}
vm_obj torch_round(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::round(to_torch(x)));}
vm_obj torch_rsqrt(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::rsqrt(to_torch(x)));}
vm_obj torch_sign(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::sign(to_torch(x)));}
vm_obj torch_sin(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::sin(to_torch(x)));}
vm_obj torch_sinh(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::sinh(to_torch(x)));}
vm_obj torch_sqrt(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::sqrt(to_torch(x)));}
vm_obj torch_tan(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::tan(to_torch(x)));}
vm_obj torch_tanh(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::tanh(to_torch(x)));}
vm_obj torch_trunc(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::trunc(to_torch(x)));}

// reduction operations


vm_obj torch_det(vm_obj const & /* dim */, vm_obj const & x) { return to_obj(torch::det(to_torch(x))); }
vm_obj torch_trace(vm_obj const & /* dim */, vm_obj const & x) { return to_obj(torch::trace(to_torch(x))); }


// nn operators
vm_obj torch_relu(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(torch::relu(to_torch(x)));}
vm_obj torch_hardtanh(vm_obj const & /*shape*/, vm_obj const & x) { return to_obj(torch::hardtanh(to_torch(x)));}
vm_obj torch_elu(vm_obj const & /*shape*/, vm_obj const & x) { return to_obj(torch::elu(to_torch(x)));}
vm_obj torch_selu(vm_obj const & /*shape*/, vm_obj const & x) { return to_obj(torch::selu(to_torch(x)));}
vm_obj torch_celu(vm_obj const & /*shape*/, vm_obj const & x) { return to_obj(torch::celu(to_torch(x)));}
vm_obj torch_leaky_relu(vm_obj const & /*shape*/, vm_obj const & x) { return to_obj(torch::leaky_relu(to_torch(x)));}
vm_obj torch_rrelu(vm_obj const & /*shape*/, vm_obj const & x) { return to_obj(torch::rrelu(to_torch(x)));}
vm_obj torch_glu(vm_obj const & /*shape*/, vm_obj const & x) { return to_obj(torch::glu(to_torch(x)));}
vm_obj torch_gelu(vm_obj const & /*shape*/, vm_obj const & x) { return to_obj(torch::gelu(to_torch(x)));}
vm_obj torch_log_sigmoid(vm_obj const & /*shape*/, vm_obj const & x) { return to_obj(torch::log_sigmoid(to_torch(x)));}
// vm_obj torch_log_softmax(vm_obj const & /*shape*/, vm_obj const & x) { return to_obj(torch::log_softmax(to_torch(x)));}

vm_obj torch_grad(vm_obj const & /* shape */, vm_obj const & x) { return to_obj(to_torch(x).grad()); }





vm_obj torch_transpose(vm_obj const & /* m */, vm_obj const & /* n */, vm_obj const & x) { return to_obj(torch::t(to_torch(x))); }
vm_obj torch_mm(vm_obj const & /* m */, vm_obj const & /* n */, vm_obj const & /* p */, vm_obj const & x, vm_obj const & y) { return to_obj(torch::mm(to_torch(x), to_torch(y))); }
vm_obj torch_bmm(vm_obj const & /* b */, vm_obj const & /* m */, vm_obj const & /* n */, vm_obj const & /* p */, vm_obj const & x, vm_obj const & y) { return to_obj(torch::bmm(to_torch(x), to_torch(y))); }



void initialize_vm_torch() {
    DECLARE_VM_BUILTIN(name({"torch", "T"}), torch_dummy);
    DECLARE_VM_BUILTIN(name({"torch", "repr"}), torch_to_repr);
    DECLARE_VM_BUILTIN(name({"torch", "zeros"}), torch_zeros);
    DECLARE_VM_BUILTIN(name({"torch", "ones"}), torch_ones);
    DECLARE_VM_BUILTIN(name({"torch", "zeros_like"}), torch_zeros_like);
    DECLARE_VM_BUILTIN(name({"torch", "ones_like"}), torch_ones_like);

    // tensor methods
    DECLARE_VM_BUILTIN(name({"torch", "grad_of"}), grad_of);
    DECLARE_VM_BUILTIN(name({"torch", "backward"}), backward);

    //
    DECLARE_VM_BUILTIN(name({"torch", "stack"}), torch_stack);

    // random
    DECLARE_VM_BUILTIN(name({"torch", "randn"}), torch_randn);
    // mathematical operations
    DECLARE_VM_BUILTIN(name({"torch", "abs"}), torch_abs);
    DECLARE_VM_BUILTIN(name({"torch", "acos"}), torch_acos);
    DECLARE_VM_BUILTIN(name({"torch", "add"}), torch_add);
    DECLARE_VM_BUILTIN(name({"torch", "asin"}), torch_asin);
    DECLARE_VM_BUILTIN(name({"torch", "atan"}), torch_atan);
    DECLARE_VM_BUILTIN(name({"torch", "bitwise_not"}), torch_bitwise_not);
    DECLARE_VM_BUILTIN(name({"torch", "ceil"}), torch_ceil);
    DECLARE_VM_BUILTIN(name({"torch", "cos"}), torch_cos);
    DECLARE_VM_BUILTIN(name({"torch", "cosh"}), torch_cosh);
    DECLARE_VM_BUILTIN(name({"torch", "div"}), torch_div);
    DECLARE_VM_BUILTIN(name({"torch", "digamma"}), torch_digamma);
    DECLARE_VM_BUILTIN(name({"torch", "erf"}), torch_erf);
    DECLARE_VM_BUILTIN(name({"torch", "erfc"}), torch_erfc);
    DECLARE_VM_BUILTIN(name({"torch", "erfinv"}), torch_erfinv);
    DECLARE_VM_BUILTIN(name({"torch", "exp"}), torch_exp);
    DECLARE_VM_BUILTIN(name({"torch", "expm1"}), torch_expm1);
    DECLARE_VM_BUILTIN(name({"torch", "floor"}), torch_floor);
    DECLARE_VM_BUILTIN(name({"torch", "fmod"}), torch_fmod);
    DECLARE_VM_BUILTIN(name({"torch", "frac"}), torch_frac);
//    DECLARE_VM_BUILTIN(name({"torch", "lerp"}), torch_lerp);
    DECLARE_VM_BUILTIN(name({"torch", "lgamma"}), torch_lgamma);
    DECLARE_VM_BUILTIN(name({"torch", "log"}), torch_log);
    DECLARE_VM_BUILTIN(name({"torch", "log10"}), torch_log10);
    DECLARE_VM_BUILTIN(name({"torch", "log1p"}), torch_log1p);
    DECLARE_VM_BUILTIN(name({"torch", "log2"}), torch_log2);
    DECLARE_VM_BUILTIN(name({"torch", "logical_not"}), torch_logical_not);
    DECLARE_VM_BUILTIN(name({"torch", "logical_xor"}), torch_logical_xor);
    DECLARE_VM_BUILTIN(name({"torch", "mul"}), torch_mul);
    DECLARE_VM_BUILTIN(name({"torch", "neg"}), torch_neg);
//    DECLARE_VM_BUILTIN(name({"torch", "pow"}), torch_pow);
    DECLARE_VM_BUILTIN(name({"torch", "reciprocal"}), torch_reciprocal);
//    DECLARE_VM_BUILTIN(name({"torch", "remainder"}), torch_remainder);
    DECLARE_VM_BUILTIN(name({"torch", "round"}), torch_round);
    DECLARE_VM_BUILTIN(name({"torch", "rsqrt"}), torch_rsqrt);
    DECLARE_VM_BUILTIN(name({"torch", "sign"}), torch_sign);
    DECLARE_VM_BUILTIN(name({"torch", "sin"}), torch_sin);
    DECLARE_VM_BUILTIN(name({"torch", "sinh"}), torch_sinh);
    DECLARE_VM_BUILTIN(name({"torch", "sqrt"}), torch_sqrt);
    DECLARE_VM_BUILTIN(name({"torch", "tan"}), torch_tan);
    DECLARE_VM_BUILTIN(name({"torch", "tanh"}), torch_tanh);
    DECLARE_VM_BUILTIN(name({"torch", "trunc"}), torch_trunc);


    DECLARE_VM_BUILTIN(name({"torch", "mm"}), torch_mm);
    DECLARE_VM_BUILTIN(name({"torch", "bmm"}), torch_bmm);

    // neural network operations
    DECLARE_VM_BUILTIN(name({"torch", "nn", "relu"}), torch_relu);
    DECLARE_VM_BUILTIN(name({"torch", "nn", "hardtanh"}), torch_hardtanh);
    DECLARE_VM_BUILTIN(name({"torch", "nn", "elu"}), torch_elu);
    DECLARE_VM_BUILTIN(name({"torch", "nn", "selu"}), torch_selu);
    DECLARE_VM_BUILTIN(name({"torch", "nn", "celu"}), torch_celu);
    DECLARE_VM_BUILTIN(name({"torch", "nn", "leaky_relu"}), torch_leaky_relu);
    DECLARE_VM_BUILTIN(name({"torch", "nn", "rrelu"}), torch_rrelu);
    DECLARE_VM_BUILTIN(name({"torch", "nn", "glu"}), torch_glu);
    DECLARE_VM_BUILTIN(name({"torch", "nn", "gelu"}), torch_gelu);
    DECLARE_VM_BUILTIN(name({"torch", "nn", "log_sigmoid"}), torch_log_sigmoid);
//    DECLARE_VM_BUILTIN(name({"torch", "nn", "log_softmax"}), torch_log_softmax);


    // autograd primitives
    DECLARE_VM_BUILTIN(name({"torch", "autograd", "grad"}), grad);

    DECLARE_VM_BUILTIN(name({"torch", "det"}), torch_det);
    DECLARE_VM_BUILTIN(name({"torch", "trace"}), torch_trace);


    // there is two: torch.t torch.transpose
    DECLARE_VM_BUILTIN(name({"torch", "transpose"}), torch_transpose);
}
void finalize_vm_torch() {
}
}