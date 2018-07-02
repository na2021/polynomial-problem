//! [Rust][1] simulations using input/output examples to learn [typed][2] first-order [term rewriting systems][3] that perform list routines.
//!
//! [1]: https://www.rust-lang.org
//! "The Rust Programming Language"
//! [2]: https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! "Wikipedia - Hindley-Milner Type System"
//! [3]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
//! "Wikipedia - Term Rewriting Systems"
#[macro_use]
extern crate polytype;
extern crate programinduction;
extern crate rand;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate term_rewriting;

use std::process::Command;
use term_rewriting::{Rule, Signature};
use std::str;
use polytype::{Context as TypeContext, TypeSchema};
use programinduction::{GP, GPParams};
use programinduction::trs::{make_task_from_data, Lexicon,  GeneticParams, ModelParams};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use utils::*;

fn main() {
    // initialize prng
    let rng = &mut SmallRng::from_seed([1u8; 16]);

    // initialize parameters
    let n_data = 10;
    let exe_home = "/Users/rule/sync/josh/library/research/list-routines/list-routines-static";
    let p_partial = 0.2;
    let p_observe = 0.0;
    let max_steps = 50;
    let max_size = Some(500);
    let n_crosses = 5;
    let p_add = 0.5;
    let p_keep = 0.5;
    let max_sample_depth = 4;
    let generations = 25;
    let population_size = 20;
    let tournament_size = 5;
    let mutation_prob = 0.75;
    let n_delta = 15;

    // initialize structs
    let knowledge: Vec<Rule> = vec![];
    let mut ctx = TypeContext::default();
    let mut sig = Signature::default();
    let mut ops: Vec<TypeSchema> = vec![];
    let vars: Vec<TypeSchema> = vec![];

    start_section("Choosing Routine");
    let routine = loop {
        let output = Command::new(exe_home)
            .arg("-u")
            .arg("--routines")
            .arg("1")
            .arg("--examples")
            .arg(&n_data.to_string())
            .output().unwrap();
        let routines_str = str::from_utf8(&output.stdout).unwrap();
        let routines: Vec<Routine> = serde_json::from_str(routines_str).unwrap();
        match routines.len() {
            0 => (),
            1 => break routines[0].clone(),
            _ => panic!("static exporter returned more than one routine.")
        }
    };

    start_section("Concept");
    let otp = TypeSchema::from(routine.tp.o);
    let itp = TypeSchema::from(routine.tp.i);
    let tp = ptp!(@arrow[itp.instantiate(&mut ctx), otp.instantiate(&mut ctx)]);
    let concept = sig.new_op(1, Some("C".to_string()));
    ops.push(tp.clone());
    println!("Name/Arity: {}/{}", concept.display(&sig), concept.arity(&sig));
    println!("Definition: {}", routine.name);
    println!("Type: {}", tp);

    start_section("Data");
    let data: Vec<Rule> = routine.examples.into_iter().map(|x| x.to_rule(&mut sig, &mut ops, concept)).collect();
    for datum in &data {
        println!("{}", datum.pretty(&sig));
    }

    start_section("Initializing Population");
    let model_params = ModelParams {
        p_partial,
        p_observe,
        max_steps,
        max_size,
    };
    let gp_params = GPParams {
        population_size,
        tournament_size,
        mutation_prob,
        n_delta,
    };
    let params = GeneticParams { max_sample_depth, n_crosses, p_add, p_keep };
    let task = make_task_from_data(&data, otp, model_params);
    let s = Lexicon::from_signature(sig, ops, vars, knowledge);
    let mut pop = s.init(&params, rng, &gp_params, &task);

    start_section("Evolving");
    for i in 0..generations {
        println!("{}...", i);
        s.evolve(&params, rng, &gp_params, &task, &mut pop);
        for (i, (individual, score)) in pop.iter().enumerate() {
            println!("{}: {}", i, score);
            println!("    {}", individual);
        }
    }

    start_section("Results");
    for (i, (individual, score)) in pop.iter().enumerate() {
        println!("{}: {}", i, score);
        println!("    {}", individual);
    }
}

fn start_section(s: &str) {
    println!("\n{}\n{}", s, "-".repeat(s.len()));
}

mod utils {
    use term_rewriting::{Operator, Rule, Signature, Term};
    use polytype::TypeSchema;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Routine {
        #[serde(rename="type")]
        pub tp: RoutineType,
        pub examples: Vec<Datum>,
        pub name: String,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Datum {
        i: Value,
        o: Value
    }
    impl Datum {
        /// Convert a `Datum` to a term rewriting [`Rule`].
        ///
        /// [`Rule`]: ../term_rewriting/struct.Rule.html
        pub fn to_rule(
            &self,
            sig: &mut Signature,
            ops: &mut Vec<TypeSchema>,
            concept: Operator
        ) -> Rule {
            let lhs = self.i.to_term(sig, ops, Some(concept));
            let rhs = self.o.to_term(sig, ops, None);
            Rule::new(lhs, vec![rhs]).unwrap()
        }
    }

    #[derive(Copy, Clone, Debug, Serialize, Deserialize)]
    pub struct RoutineType {
        #[serde(rename="input")]
        pub i: IOType,
        #[serde(rename="output")]
        pub o: IOType
    }

    #[derive(Copy, Clone, Debug, Serialize, Deserialize)]
    pub enum IOType {
        #[serde(rename="bool")]
        Bool,
        #[serde(rename="list-of-int")]
        IntList,
        #[serde(rename="int")]
        Int,
    }
    impl From<IOType> for TypeSchema {
        fn from(t: IOType) -> Self {
            match t {
                IOType::Bool => ptp!(bool),
                IOType::Int => ptp!(int),
                IOType::IntList => ptp!(list(tp!(int))),
            }
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(untagged)]
    pub enum Value {
        Int(isize),
        IntList(Vec<isize>),
        Bool(bool),
    }
    impl Value {
        fn to_term(
            &self,
            sig: &mut Signature,
            ops: &mut Vec<TypeSchema>,
            lhs: Option<Operator>,
        ) -> Term {
            let base_term = match self {
                Value::Int(x) => Value::num_to_term(sig, ops, *x),
                Value::IntList(xs) => Value::list_to_term(sig, ops, xs),
                Value::Bool(true) => Term::Application{
                    op: Value::get_op(sig, ops, 0, Some("true"), &ptp!(bool)),
                    args: vec![]
                },
                Value::Bool(false) => Term::Application{
                    op: Value::get_op(sig, ops, 0, Some("false"), &ptp!(bool)),
                    args: vec![]
                },
            };
            if let Some(op) = lhs {
                Term::Application {
                    op,
                    args: vec![base_term]
                }
            } else {
                base_term
            }
        }
        fn list_to_term(
            sig: &mut Signature,
            ops: &mut Vec<TypeSchema>,
            xs: &[isize],
        ) -> Term {
            let ts: Vec<Term> = xs.iter().map(|&x| Value::num_to_term(sig, ops, x)).rev().collect();
            let nil = Value::get_op(sig, ops, 0, Some("nil"), &ptp!(0; list(tp!(0))));
            let cons = Value::get_op(sig, ops, 2, Some("cons"), &ptp!(0; @arrow[tp!(0), tp!(list(tp!(0)))]));
            let mut term = Term::Application{ op: nil, args: vec![] };
            for t in ts {
                term = Term::Application {
                    op: cons,
                    args: vec![t, term]
                };
            }
            term
        }
        fn num_to_term(
            sig: &mut Signature,
            ops: &mut Vec<TypeSchema>,
            n: isize,
        ) -> Term {
            if n < 0 {
                let neg = Value::get_op(sig, ops, 1, Some("neg"), &ptp!(@arrow[tp!(int), tp!(int)]));
                let num = Value::get_op(sig, ops, 0, Some(&(-n).to_string()), &ptp!(int));
                Term::Application {
                    op: neg,
                    args: vec![Term::Application {
                        op: num,
                        args: vec![]
                    }],
                }
            } else {
                let num = Value::get_op(sig, ops, 0, Some(&n.to_string()), &ptp!(int));
                Term::Application {
                    op: num,
                    args: vec![],
                }
            }
        }
        fn get_op(sig: &mut Signature, ops: &mut Vec<TypeSchema>, arity: u32, name: Option<&str>, schema: &TypeSchema) -> Operator {
            if let Some(op) = Value::find_op(sig, ops, arity, name, schema) {
                op
            } else {
                ops.push(schema.clone());
                sig.new_op(arity, name.map(|x| x.to_string()))
            }
        }
        fn find_op(sig: &Signature, ops: &mut Vec<TypeSchema>, arity: u32, name: Option<&str>, schema: &TypeSchema) -> Option<Operator> {
            for (i, o) in sig.operators().into_iter().enumerate() {
                if o.arity(&sig) == arity && o.name(&sig) == name && ops[i] == *schema {
                    return Some(o);
                }
            }
            None
        }
    }
}
