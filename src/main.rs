#[macro_use]
extern crate polytype;
extern crate programinduction;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate term_rewriting;

use std::process::Command;
use term_rewriting::{Signature, Rule};
use std::str;
use utils::*;
use polytype::TypeSchema;
use programinduction::trs::make_task_from_data;

fn main() {

    // basic parameters;
    let n_data = 10;
    let exe_home = "/Users/rule/sync/josh/library/research/list-routines/list-routines-static";
    let p_partial = 0.0;
    let prior_temperature = 1.0;
    let likelihood_temperature = 1.0;
    let temperature = 1.0;


    // sample a routine and N data points
    let output = Command::new(exe_home)
        .arg("-u")
        .arg("--routines")
        .arg("1")
        .arg("--examples")
        .arg(&n_data.to_string())
        .output().unwrap();

    // parse the routine into a string
    let routines_str = str::from_utf8(&output.stdout).unwrap();

    // parse the string into a Routine
    let routines: Vec<Routine> = serde_json::from_str(routines_str).unwrap();
    let routine = routines[0].clone();

    // construct the Type
    let tp = TypeSchema::from(routine.tp.o);

    // convert the data points to rules
    let (mut sig, _) = Signature::new(vec![(1, Some("C".to_string()))]);
    let data: Vec<Rule> = routine.examples.into_iter().map(|x| x.to_rule(&mut sig)).collect();

    // construct the task
    let task = make_task_from_data(
        &data,
        tp,
        p_partial,
        temperature,
        prior_temperature,
        likelihood_temperature,
    );
    // construct the GP params
    // construct the initial TRS
    // run the algorithm
    // report the results
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
        /// Convert a `self` to a term rewriting [`Rule`].
        ///
        /// [`Rule`]: ../term_rewriting/struct.Rule.html
        pub fn to_rule(&self, sig: &mut Signature) -> Rule {
            let lhs = self.i.to_term(sig, Some("C"));
            let rhs = self.o.to_term(sig, None);
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
        fn to_term(&self, sig: &mut Signature, lhs: Option<&str>) -> Term {
            let base_term = match self {
                Value::Int(x) => Value::num_to_term(*x, sig),
                Value::IntList(xs) => Value::list_to_term(xs, sig),
                Value::Bool(true) => Term::Application{
                    op: Value::get_op(sig, 0, Some("true")),
                    args: vec![]
                },
                Value::Bool(false) => Term::Application{
                    op: Value::get_op(sig, 0, Some("false")),
                    args: vec![]
                },
            };
            match lhs {
                Some(lhs_name) => {
                    let op = Value::get_op(sig, 1, Some(lhs_name));
                    Term::Application {
                        op,
                        args: vec![base_term]
                    }
                }
                None => base_term
            }
        }
        fn list_to_term(xs: &[isize], sig: &mut Signature) -> Term {
            let ts: Vec<Term> = xs.iter().map(|&x| Value::num_to_term(x, sig)).rev().collect();
            let nil = Value::get_op(sig, 0, Some("nil"));
            let cons = Value::get_op(sig, 2, Some("cons"));
            let mut term = Term::Application{ op: nil, args: vec![] };
            for t in ts {
                term = Term::Application {
                    op: cons,
                    args: vec![t, term]
                };
            }
            term
        }
        fn num_to_term(n: isize, sig: &mut Signature) -> Term {
            if n < 0 {
                let neg = Value::get_op(sig, 1, Some("neg"));
                let num = Value::get_op(sig, 0, Some(&(-n).to_string()));
                Term::Application {
                    op: neg,
                    args: vec![Term::Application {
                        op: num,
                        args: vec![]
                    }],
                }
            } else {
                let num = Value::get_op(sig, 0, Some(&n.to_string()));
                Term::Application {
                    op: num,
                    args: vec![],
                }
            }
        }
        fn get_op(sig: &mut Signature, arity: u32, name: Option<&str>) -> Operator {
            if let Some(op) = Value::find_op(sig, arity, name) {
                op
            } else {
                sig.new_op(arity, name.map(|x| x.to_string()))
            }
        }
        fn find_op(sig: &Signature, arity: u32, name: Option<&str>) -> Option<Operator> {
            for o in sig.operators() {
                if o.arity(sig) == arity && o.name(sig) == name {
                    return Some(o);
                }
            }
            None
        }
    }
}
