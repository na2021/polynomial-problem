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
use term_rewriting::Rule;
use std::str;
use polytype::TypeSchema;
use programinduction::{GP, GPParams};
use programinduction::trs::{make_task_from_data, TRS, TRSParams, TRSSpace};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use utils::*;

fn main() {
    // seed the prng
    let rng = &mut SmallRng::from_seed([1u8; 16]);

    // basic parameters;
    let n_data = 10;
    let exe_home = "/Users/rule/sync/josh/library/research/list-routines/list-routines-static";
    let p_partial = 0.0;
    let prior_temperature = 1.0;
    let likelihood_temperature = 1.0;
    let temperature = 1.0;
    let generations = 10;
    let n_crosses = 50;
    let population_size = 10;
    let tournament_size = 5;
    let mutation_prob = 0.6;
    let n_delta = 1;

    // sample a routine and N data points
    let output = Command::new(exe_home)
        .arg("-u")
        .arg("--routines")
        .arg("1")
        .arg("--examples")
        .arg(&n_data.to_string())
        .output().unwrap();
    let routines_str = str::from_utf8(&output.stdout).unwrap();
    let routines: Vec<Routine> = serde_json::from_str(routines_str).unwrap();
    let routine = routines[0].clone();

    // initialize background knowledge
    let mut h0 = TRS::default();

    // construct the Type
    let otp = TypeSchema::from(routine.tp.o);
    let itp = TypeSchema::from(routine.tp.i);
    let tp = ptp!(@arrow[otp.instantiate(&mut h0.ctx), itp.instantiate(&mut h0.ctx)]);
    let concept = h0.signature.new_op(1, Some("C".to_string()));
    h0.ops.push(tp);

    // convert the data points to rules
    let data: Vec<Rule> = routine.examples.into_iter().map(|x| x.to_rule(&mut h0, concept)).collect();

    // construct the task
    let task = make_task_from_data(
        &data,
        otp,
        p_partial,
        temperature,
        prior_temperature,
        likelihood_temperature,
    );

    // construct the params
    let gpparams = GPParams {
        population_size,
        tournament_size,
        mutation_prob,
        n_delta,
    };
    let params = TRSParams { h0, n_crosses };
    let s = TRSSpace;

    // run the algorithm
    let mut pop = s.init(&params, rng, &gpparams, &task);
    for _ in 0..generations {
        s.evolve(&params, rng, &gpparams, &task, &mut pop)
    }

    // report the results
}

mod utils {
    use term_rewriting::{Operator, Rule, Term};
    use polytype::TypeSchema;
    use programinduction::trs::TRS;

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
        /// Convert a `self` to a term rewriting [`Rule`].
        ///
        /// [`Rule`]: ../term_rewriting/struct.Rule.html
        pub fn to_rule(&self, trs: &mut TRS, concept: Operator) -> Rule {
            let lhs = self.i.to_term(trs, Some(concept));
            let rhs = self.o.to_term(trs, None);
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
        fn to_term(&self, trs: &mut TRS, lhs: Option<Operator>) -> Term {
            let base_term = match self {
                Value::Int(x) => Value::num_to_term(trs, *x),
                Value::IntList(xs) => Value::list_to_term(trs, xs),
                Value::Bool(true) => Term::Application{
                    op: Value::get_op(trs, 0, Some("true"), &ptp!(bool)),
                    args: vec![]
                },
                Value::Bool(false) => Term::Application{
                    op: Value::get_op(trs, 0, Some("false"), &ptp!(bool)),
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
        fn list_to_term(trs: &mut TRS, xs: &[isize]) -> Term {
            let ts: Vec<Term> = xs.iter().map(|&x| Value::num_to_term(trs, x)).rev().collect();
            let nil = Value::get_op(trs, 0, Some("nil"), &ptp!(0; list(tp!(0))));
            let cons = Value::get_op(trs, 2, Some("cons"), &ptp!(0; @arrow[tp!(0), tp!(list(tp!(0)))]));
            let mut term = Term::Application{ op: nil, args: vec![] };
            for t in ts {
                term = Term::Application {
                    op: cons,
                    args: vec![t, term]
                };
            }
            term
        }
        fn num_to_term(trs: &mut TRS, n: isize) -> Term {
            if n < 0 {
                let neg = Value::get_op(trs, 1, Some("neg"), &ptp!(@arrow[tp!(int), tp!(int)]));
                let num = Value::get_op(trs, 0, Some(&(-n).to_string()), &ptp!(int));
                Term::Application {
                    op: neg,
                    args: vec![Term::Application {
                        op: num,
                        args: vec![]
                    }],
                }
            } else {
                let num = Value::get_op(trs, 0, Some(&n.to_string()), &ptp!(int));
                Term::Application {
                    op: num,
                    args: vec![],
                }
            }
        }
        fn get_op(trs: &mut TRS, arity: u32, name: Option<&str>, schema: &TypeSchema) -> Operator {
            if let Some(op) = Value::find_op(trs, arity, name, schema) {
                op
            } else {
                trs.ops.push(schema.clone());
                trs.signature.new_op(arity, name.map(|x| x.to_string()))
            }
        }
        fn find_op(trs: &TRS, arity: u32, name: Option<&str>, schema: &TypeSchema) -> Option<Operator> {
            for (i, o) in trs.signature.operators().into_iter().enumerate() {
                if o.arity(&trs.signature) == arity && o.name(&trs.signature) == name && trs.ops[i] == *schema {
                    return Some(o);
                }
            }
            None
        }
    }
}
