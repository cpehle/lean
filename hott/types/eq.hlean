/-
Copyright (c) 2014 Floris van Doorn. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Author: Floris van Doorn

Partially ported from Coq HoTT
Theorems about path types (identity types)
-/

open eq sigma sigma.ops equiv is_equiv equiv.ops

-- TODO: Rename transport_eq_... and pathover_eq_... to eq_transport_... and eq_pathover_...

namespace eq
  /- Path spaces -/

  variables {A B : Type} {a a1 a2 a3 a4 : A} {b b1 b2 : B} {f g : A → B} {h : B → A}
            {p p' p'' : a1 = a2}

  /- The path spaces of a path space are not, of course, determined; they are just the
      higher-dimensional structure of the original space. -/

  /- some lemmas about whiskering or other higher paths -/

  theorem whisker_left_con_right (p : a1 = a2) {q q' q'' : a2 = a3} (r : q = q') (s : q' = q'')
    : whisker_left p (r ⬝ s) = whisker_left p r ⬝ whisker_left p s :=
  begin
    induction p, induction r, induction s, reflexivity
  end

  theorem whisker_right_con_right (q : a2 = a3) (r : p = p') (s : p' = p'')
    : whisker_right (r ⬝ s) q = whisker_right r q ⬝ whisker_right s q :=
  begin
    induction q, induction r, induction s, reflexivity
  end

  theorem whisker_left_con_left (p : a1 = a2) (p' : a2 = a3) {q q' : a3 = a4} (r : q = q')
    : whisker_left (p ⬝ p') r = !con.assoc ⬝ whisker_left p (whisker_left p' r) ⬝ !con.assoc' :=
  begin
    induction p', induction p, induction r, induction q, reflexivity
  end

  theorem whisker_right_con_left {p p' : a1 = a2} (q : a2 = a3) (q' : a3 = a4) (r : p = p')
    : whisker_right r (q ⬝ q') = !con.assoc' ⬝ whisker_right (whisker_right r q) q' ⬝ !con.assoc :=
  begin
    induction q', induction q, induction r, induction p, reflexivity
  end

  theorem whisker_left_inv_left (p : a2 = a1) {q q' : a2 = a3} (r : q = q')
    : !con_inv_cancel_left⁻¹ ⬝ whisker_left p (whisker_left p⁻¹ r) ⬝ !con_inv_cancel_left = r :=
  begin
    induction p, induction r, induction q, reflexivity
  end

  theorem whisker_left_inv (p : a1 = a2) {q q' : a2 = a3} (r : q = q')
    : whisker_left p r⁻¹ = (whisker_left p r)⁻¹ :=
  by induction r; reflexivity

  theorem whisker_right_inv {p p' : a1 = a2} (q : a2 = a3) (r : p = p')
    : whisker_right r⁻¹ q = (whisker_right r q)⁻¹ :=
  by induction r; reflexivity

  theorem ap_eq_ap10 {f g : A → B} (p : f = g) (a : A) : ap (λh, h a) p = ap10 p a :=
  by induction p;reflexivity

  theorem inverse2_right_inv (r : p = p') : r ◾ inverse2 r ⬝ con.right_inv p' = con.right_inv p :=
  by induction r;induction p;reflexivity

  theorem inverse2_left_inv (r : p = p') : inverse2 r ◾ r ⬝ con.left_inv p' = con.left_inv p :=
  by induction r;induction p;reflexivity

  theorem ap_con_right_inv (f : A → B) (p : a1 = a2)
    : ap_con f p p⁻¹ ⬝ whisker_left _ (ap_inv f p) ⬝ con.right_inv (ap f p)
      = ap (ap f) (con.right_inv p) :=
  by induction p;reflexivity

  theorem ap_con_left_inv (f : A → B) (p : a1 = a2)
    : ap_con f p⁻¹ p ⬝ whisker_right (ap_inv f p) _ ⬝ con.left_inv (ap f p)
      = ap (ap f) (con.left_inv p) :=
  by induction p;reflexivity

  theorem idp_con_whisker_left {q q' : a2 = a3} (r : q = q') :
  !idp_con⁻¹ ⬝ whisker_left idp r = r ⬝ !idp_con⁻¹ :=
  by induction r;induction q;reflexivity

  theorem whisker_left_idp_con {q q' : a2 = a3} (r : q = q') :
  whisker_left idp r ⬝ !idp_con = !idp_con ⬝ r :=
  by induction r;induction q;reflexivity

  theorem idp_con_idp {p : a = a} (q : p = idp) : idp_con p ⬝ q = ap (λp, idp ⬝ p) q :=
  by cases q;reflexivity

  /- Transporting in path spaces.

     There are potentially a lot of these lemmas, so we adopt a uniform naming scheme:

     - `l` means the left endpoint varies
     - `r` means the right endpoint varies
     - `F` means application of a function to that (varying) endpoint.
  -/

  definition transport_eq_l (p : a1 = a2) (q : a1 = a3)
    : transport (λx, x = a3) p q = p⁻¹ ⬝ q :=
  by induction p; induction q; reflexivity

  definition transport_eq_r (p : a2 = a3) (q : a1 = a2)
    : transport (λx, a1 = x) p q = q ⬝ p :=
  by induction p; induction q; reflexivity

  definition transport_eq_lr (p : a1 = a2) (q : a1 = a1)
    : transport (λx, x = x) p q = p⁻¹ ⬝ q ⬝ p :=
  by induction p; rewrite [▸*,idp_con]

  definition transport_eq_Fl (p : a1 = a2) (q : f a1 = b)
    : transport (λx, f x = b) p q = (ap f p)⁻¹ ⬝ q :=
  by induction p; induction q; reflexivity

  definition transport_eq_Fr (p : a1 = a2) (q : b = f a1)
    : transport (λx, b = f x) p q = q ⬝ (ap f p) :=
  by induction p; reflexivity

  definition transport_eq_FlFr (p : a1 = a2) (q : f a1 = g a1)
    : transport (λx, f x = g x) p q = (ap f p)⁻¹ ⬝ q ⬝ (ap g p) :=
  by induction p; rewrite [▸*,idp_con]

  definition transport_eq_FlFr_D {B : A → Type} {f g : Πa, B a}
    (p : a1 = a2) (q : f a1 = g a1)
      : transport (λx, f x = g x) p q = (apd f p)⁻¹ ⬝ ap (transport B p) q ⬝ (apd g p) :=
  by induction p; rewrite [▸*,idp_con,ap_id]

  definition transport_eq_FFlr (p : a1 = a2) (q : h (f a1) = a1)
    : transport (λx, h (f x) = x) p q = (ap h (ap f p))⁻¹ ⬝ q ⬝ p :=
  by induction p; rewrite [▸*,idp_con]

  definition transport_eq_lFFr (p : a1 = a2) (q : a1 = h (f a1))
    : transport (λx, x = h (f x)) p q = p⁻¹ ⬝ q ⬝ (ap h (ap f p)) :=
  by induction p; rewrite [▸*,idp_con]

  /- Pathovers -/

  -- In the comment we give the fibration of the pathover

  -- we should probably try to do everything just with pathover_eq (defined in cubical.square),
  -- the following definitions may be removed in future.

  definition pathover_eq_l (p : a1 = a2) (q : a1 = a3) : q =[p] p⁻¹ ⬝ q := /-(λx, x = a3)-/
  by induction p; induction q; exact idpo

  definition pathover_eq_r (p : a2 = a3) (q : a1 = a2) : q =[p] q ⬝ p := /-(λx, a1 = x)-/
  by induction p; induction q; exact idpo

  definition pathover_eq_lr (p : a1 = a2) (q : a1 = a1) : q =[p] p⁻¹ ⬝ q ⬝ p := /-(λx, x = x)-/
  by induction p; rewrite [▸*,idp_con]; exact idpo

  definition pathover_eq_Fl (p : a1 = a2) (q : f a1 = b) : q =[p] (ap f p)⁻¹ ⬝ q := /-(λx, f x = b)-/
  by induction p; induction q; exact idpo

  definition pathover_eq_Fr (p : a1 = a2) (q : b = f a1) : q =[p] q ⬝ (ap f p) := /-(λx, b = f x)-/
  by induction p; exact idpo

  definition pathover_eq_FlFr (p : a1 = a2) (q : f a1 = g a1) : q =[p] (ap f p)⁻¹ ⬝ q ⬝ (ap g p) :=
  /-(λx, f x = g x)-/
  by induction p; rewrite [▸*,idp_con]; exact idpo

  definition pathover_eq_FlFr_D {B : A → Type} {f g : Πa, B a} (p : a1 = a2) (q : f a1 = g a1)
    : q =[p] (apd f p)⁻¹ ⬝ ap (transport B p) q ⬝ (apd g p) := /-(λx, f x = g x)-/
  by induction p; rewrite [▸*,idp_con,ap_id];exact idpo

  definition pathover_eq_FFlr (p : a1 = a2) (q : h (f a1) = a1) : q =[p] (ap h (ap f p))⁻¹ ⬝ q ⬝ p :=
  /-(λx, h (f x) = x)-/
  by induction p; rewrite [▸*,idp_con];exact idpo

  definition pathover_eq_lFFr (p : a1 = a2) (q : a1 = h (f a1)) : q =[p] p⁻¹ ⬝ q ⬝ (ap h (ap f p)) :=
  /-(λx, x = h (f x))-/
  by induction p; rewrite [▸*,idp_con];exact idpo

  definition pathover_eq_r_idp (p : a1 = a2) : idp =[p] p := /-(λx, a1 = x)-/
  by induction p; exact idpo

  definition pathover_eq_l_idp (p : a1 = a2) : idp =[p] p⁻¹ := /-(λx, x = a1)-/
  by induction p; exact idpo

  definition pathover_eq_l_idp' (p : a1 = a2) : idp =[p⁻¹] p := /-(λx, x = a2)-/
  by induction p; exact idpo

  -- The Functorial action of paths is [ap].

  /- Equivalences between path spaces -/

  /- [ap_closed] is in init.equiv  -/

  definition equiv_ap (f : A → B) [H : is_equiv f] (a1 a2 : A)
    : (a1 = a2) ≃ (f a1 = f a2) :=
  equiv.mk (ap f) _

  /- Path operations are equivalences -/

  definition is_equiv_eq_inverse (a1 a2 : A) : is_equiv (@inverse A a1 a2) :=
  is_equiv.mk inverse inverse inv_inv inv_inv (λp, eq.rec_on p idp)
  local attribute is_equiv_eq_inverse [instance]

  definition eq_equiv_eq_symm (a1 a2 : A) : (a1 = a2) ≃ (a2 = a1) :=
  equiv.mk inverse _

  definition is_equiv_concat_left [constructor] [instance] (p : a1 = a2) (a3 : A)
    : is_equiv (concat p : a2 = a3 → a1 = a3) :=
  is_equiv.mk (concat p) (concat p⁻¹)
              (con_inv_cancel_left p)
              (inv_con_cancel_left p)
              (λq, by induction p;induction q;reflexivity)
  local attribute is_equiv_concat_left [instance]

  definition equiv_eq_closed_left [constructor] (a3 : A) (p : a1 = a2) : (a1 = a3) ≃ (a2 = a3) :=
  equiv.mk (concat p⁻¹) _

  definition is_equiv_concat_right [constructor] [instance] (p : a2 = a3) (a1 : A)
    : is_equiv (λq : a1 = a2, q ⬝ p) :=
  is_equiv.mk (λq, q ⬝ p) (λq, q ⬝ p⁻¹)
              (λq, inv_con_cancel_right q p)
              (λq, con_inv_cancel_right q p)
              (λq, by induction p;induction q;reflexivity)
  local attribute is_equiv_concat_right [instance]

  definition equiv_eq_closed_right [constructor] (a1 : A) (p : a2 = a3) : (a1 = a2) ≃ (a1 = a3) :=
  equiv.mk (λq, q ⬝ p) _

  definition eq_equiv_eq_closed [constructor] (p : a1 = a2) (q : a3 = a4) : (a1 = a3) ≃ (a2 = a4) :=
  equiv.trans (equiv_eq_closed_left a3 p) (equiv_eq_closed_right a2 q)

  definition is_equiv_whisker_left (p : a1 = a2) (q r : a2 = a3)
  : is_equiv (@whisker_left A a1 a2 a3 p q r) :=
  begin
  fapply adjointify,
    {intro s, apply (!cancel_left s)},
    {intro s,
      apply concat, {apply whisker_left_con_right},
      apply concat, rotate_left 1, apply (whisker_left_inv_left p s),
      apply concat2,
        {apply concat, {apply whisker_left_con_right},
          apply concat2,
            {induction p, induction q, reflexivity},
            {reflexivity}},
        {induction p, induction r, reflexivity}},
    {intro s, induction s, induction q, induction p, reflexivity}
  end

  definition eq_equiv_con_eq_con_left (p : a1 = a2) (q r : a2 = a3) : (q = r) ≃ (p ⬝ q = p ⬝ r) :=
  equiv.mk _ !is_equiv_whisker_left

  definition is_equiv_whisker_right {p q : a1 = a2} (r : a2 = a3)
    : is_equiv (λs, @whisker_right A a1 a2 a3 p q s r) :=
  begin
  fapply adjointify,
    {intro s, apply (!cancel_right s)},
    {intro s, induction r, cases s, induction q, reflexivity},
    {intro s, induction s, induction r, induction p, reflexivity}
  end

  definition eq_equiv_con_eq_con_right (p q : a1 = a2) (r : a2 = a3) : (p = q) ≃ (p ⬝ r = q ⬝ r) :=
  equiv.mk _ !is_equiv_whisker_right

  /-
    The following proofs can be simplified a bit by concatenating previous equivalences.
    However, these proofs have the advantage that the inverse is definitionally equal to
    what we would expect
  -/
  definition is_equiv_con_eq_of_eq_inv_con (p : a1 = a3) (q : a2 = a3) (r : a2 = a1)
    : is_equiv (con_eq_of_eq_inv_con : p = r⁻¹ ⬝ q → r ⬝ p = q) :=
  begin
    fapply adjointify,
    { apply eq_inv_con_of_con_eq},
    { intro s, induction r, rewrite [↑[con_eq_of_eq_inv_con,eq_inv_con_of_con_eq],
        con.assoc,con.assoc,con.left_inv,▸*,-con.assoc,con.right_inv,▸* at *,idp_con s]},
    { intro s, induction r, rewrite [↑[con_eq_of_eq_inv_con,eq_inv_con_of_con_eq],
        con.assoc,con.assoc,con.right_inv,▸*,-con.assoc,con.left_inv,▸* at *,idp_con s] },
  end

  definition eq_inv_con_equiv_con_eq (p : a1 = a3) (q : a2 = a3) (r : a2 = a1)
    : (p = r⁻¹ ⬝ q) ≃ (r ⬝ p = q) :=
  equiv.mk _ !is_equiv_con_eq_of_eq_inv_con

  definition is_equiv_con_eq_of_eq_con_inv (p : a1 = a3) (q : a2 = a3) (r : a2 = a1)
    : is_equiv (con_eq_of_eq_con_inv : r = q ⬝ p⁻¹ → r ⬝ p = q) :=
  begin
    fapply adjointify,
    { apply eq_con_inv_of_con_eq},
    { intro s, induction p, rewrite [↑[con_eq_of_eq_con_inv,eq_con_inv_of_con_eq]]},
    { intro s, induction p, rewrite [↑[con_eq_of_eq_con_inv,eq_con_inv_of_con_eq]] },
  end

  definition eq_con_inv_equiv_con_eq (p : a1 = a3) (q : a2 = a3) (r : a2 = a1)
    : (r = q ⬝ p⁻¹) ≃ (r ⬝ p = q) :=
  equiv.mk _ !is_equiv_con_eq_of_eq_con_inv

  definition is_equiv_inv_con_eq_of_eq_con (p : a1 = a3) (q : a2 = a3) (r : a1 = a2)
    : is_equiv (inv_con_eq_of_eq_con : p = r ⬝ q → r⁻¹ ⬝ p = q) :=
  begin
    fapply adjointify,
    { apply eq_con_of_inv_con_eq},
    { intro s, induction r, rewrite [↑[inv_con_eq_of_eq_con,eq_con_of_inv_con_eq],
        con.assoc,con.assoc,con.left_inv,▸*,-con.assoc,con.right_inv,▸* at *,idp_con s]},
    { intro s, induction r, rewrite [↑[inv_con_eq_of_eq_con,eq_con_of_inv_con_eq],
        con.assoc,con.assoc,con.right_inv,▸*,-con.assoc,con.left_inv,▸* at *,idp_con s] },
  end

  definition eq_con_equiv_inv_con_eq (p : a1 = a3) (q : a2 = a3) (r : a1 = a2)
    : (p = r ⬝ q) ≃ (r⁻¹ ⬝ p = q) :=
  equiv.mk _ !is_equiv_inv_con_eq_of_eq_con

  definition is_equiv_con_inv_eq_of_eq_con (p : a3 = a1) (q : a2 = a3) (r : a2 = a1)
    : is_equiv (con_inv_eq_of_eq_con : r = q ⬝ p → r ⬝ p⁻¹ = q) :=
  begin
    fapply adjointify,
    { apply eq_con_of_con_inv_eq},
    { intro s, induction p, rewrite [↑[con_inv_eq_of_eq_con,eq_con_of_con_inv_eq]]},
    { intro s, induction p, rewrite [↑[con_inv_eq_of_eq_con,eq_con_of_con_inv_eq]] },
  end

  definition eq_con_equiv_con_inv_eq (p : a3 = a1) (q : a2 = a3) (r : a2 = a1)
    : (r = q ⬝ p) ≃ (r ⬝ p⁻¹ = q) :=
   equiv.mk _ !is_equiv_con_inv_eq_of_eq_con

  local attribute is_equiv_inv_con_eq_of_eq_con
                  is_equiv_con_inv_eq_of_eq_con
                  is_equiv_con_eq_of_eq_con_inv
                  is_equiv_con_eq_of_eq_inv_con [instance]

  definition is_equiv_eq_con_of_inv_con_eq (p : a1 = a3) (q : a2 = a3) (r : a2 = a1)
    : is_equiv (eq_con_of_inv_con_eq : r⁻¹ ⬝ q = p → q = r ⬝ p) :=
  is_equiv_inv inv_con_eq_of_eq_con

  definition is_equiv_eq_con_of_con_inv_eq (p : a1 = a3) (q : a2 = a3) (r : a2 = a1)
    : is_equiv (eq_con_of_con_inv_eq : q ⬝ p⁻¹ = r → q = r ⬝ p) :=
  is_equiv_inv con_inv_eq_of_eq_con

  definition is_equiv_eq_con_inv_of_con_eq (p : a1 = a3) (q : a2 = a3) (r : a2 = a1)
    : is_equiv (eq_con_inv_of_con_eq : r ⬝ p = q → r = q ⬝ p⁻¹) :=
  is_equiv_inv con_eq_of_eq_con_inv

  definition is_equiv_eq_inv_con_of_con_eq (p : a1 = a3) (q : a2 = a3) (r : a2 = a1)
    : is_equiv (eq_inv_con_of_con_eq : r ⬝ p = q → p = r⁻¹ ⬝ q) :=
  is_equiv_inv con_eq_of_eq_inv_con

  /- Pathover Equivalences -/

  definition pathover_eq_equiv_l (p : a1 = a2) (q : a1 = a3) (r : a2 = a3) : q =[p] r ≃ q = p ⬝ r :=
  /-(λx, x = a3)-/
  by induction p; exact !pathover_idp ⬝e !equiv_eq_closed_right !idp_con⁻¹

  definition pathover_eq_equiv_r (p : a2 = a3) (q : a1 = a2) (r : a1 = a3) : q =[p] r ≃ q ⬝ p = r :=
  /-(λx, a1 = x)-/
  by induction p; apply pathover_idp

  definition pathover_eq_equiv_lr (p : a1 = a2) (q : a1 = a1) (r : a2 = a2)
    : q =[p] r ≃ q ⬝ p = p ⬝ r := /-(λx, x = x)-/
  by induction p; exact !pathover_idp ⬝e !equiv_eq_closed_right !idp_con⁻¹

  definition pathover_eq_equiv_Fl (p : a1 = a2) (q : f a1 = b) (r : f a2 = b)
    : q =[p] r ≃ q = ap f p ⬝ r := /-(λx, f x = b)-/
  by induction p; exact !pathover_idp ⬝e !equiv_eq_closed_right !idp_con⁻¹

  definition pathover_eq_equiv_Fr (p : a1 = a2) (q : b = f a1) (r : b = f a2)
    : q =[p] r ≃ q ⬝ ap f p = r := /-(λx, b = f x)-/
  by induction p; apply pathover_idp

  definition pathover_eq_equiv_FlFr (p : a1 = a2) (q : f a1 = g a1) (r : f a2 = g a2)
    : q =[p] r ≃ q ⬝ ap g p = ap f p ⬝ r := /-(λx, f x = g x)-/
  by induction p; exact !pathover_idp ⬝e !equiv_eq_closed_right !idp_con⁻¹

  definition pathover_eq_equiv_FFlr (p : a1 = a2) (q : h (f a1) = a1) (r : h (f a2) = a2)
    : q =[p] r ≃ q ⬝ p = ap h (ap f p) ⬝ r :=
  /-(λx, h (f x) = x)-/
  by induction p; exact !pathover_idp ⬝e !equiv_eq_closed_right !idp_con⁻¹

  definition pathover_eq_equiv_lFFr (p : a1 = a2) (q : a1 = h (f a1)) (r : a2 = h (f a2))
    : q =[p] r ≃ q ⬝ ap h (ap f p) = p ⬝ r :=
  /-(λx, x = h (f x))-/
  by induction p; exact !pathover_idp ⬝e !equiv_eq_closed_right !idp_con⁻¹

  -- a lot of this library still needs to be ported from Coq HoTT



end eq
