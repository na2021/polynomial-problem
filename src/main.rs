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

use polytype::{Context as TypeContext, TypeSchema};
use programinduction::{GP, GPParams};
use programinduction::trs::{make_task_from_data, Lexicon,  GeneticParams, ModelParams};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::str;
use term_rewriting::{Context, parse_trs, Rule, RuleContext, Signature};
use utils::*;

fn main() -> io::Result<()> {
    start_section("So it begins!");
    // initialize prng
    let rng = &mut SmallRng::from_seed([1u8; 16]);

    // initialize parameters
    let p_partial = 0.2;
    let p_observe = 0.0;
    let max_steps = 50;
    let max_size = Some(500);
    let n_crosses = 5;
    let p_add = 0.5;
    let p_keep = 0.5;
    let max_sample_depth = 4;
    let deterministic = true;
    let atom_weights = (0.5, 0.25, 0.25);
    let generations_per_datum = 10;
    let population_size = 5;
    let tournament_size = 5;
    let mutation_prob = 0.95;
    let n_delta = 10;

    // initialize structs
    let knowledge: Vec<Rule> = vec![];
    let mut ctx = TypeContext::default();
    let mut sig = Signature::default();
    let mut ops: Vec<TypeSchema> = vec![];
    let vars: Vec<TypeSchema> = vec![];

    start_section("Choosing Routine");
    let routine = loop {
        let routines_str = "[{\"type\":{\"input\":\"list-of-int\",\"output\":\"int\"},\"examples\":[{\"i\":[5,10],\"o\":5},{\"i\":[9,13,12],\"o\":9},{\"i\":[7,15],\"o\":7},{\"i\":[15,0,14,8,8,12],\"o\":15},{\"i\":[10,6,10,0],\"o\":10},{\"i\":[3,7,3,0],\"o\":3},{\"i\":[8,2,9],\"o\":8},{\"i\":[8,16,3,8],\"o\":8},{\"i\":[12,0,12,12,12],\"o\":12},{\"i\":[11,11,10,7,9],\"o\":11}],\"name\":\"((head (dyn . 0)))\"}]";
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
    let cons = sig.new_op(2, Some("CONS".to_string()));
    ops.push(ptp!(@arrow[tp!(int),
                         tp!(list(tp!(int))),
                         tp!(list(tp!(int)))]));
    let nil = sig.new_op(0, Some("NIL".to_string()));
    ops.push(ptp!(list(tp!(int))));
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
    let templates = vec![
        // [!] = [!]
        RuleContext {
            lhs: Context::Hole,
            rhs: vec![Context::Hole],
        },
        // C nil = [!]
        RuleContext {
            lhs: Context::Application {
                op: concept,
                args: vec![Context::Application {
                    op: nil,
                    args: vec![]
                }]
            },
            rhs: vec![Context::Hole],
        },
        // C cons([!], [!]) = [!]
        RuleContext {
            lhs: Context::Application {
                op: concept,
                args: vec![Context::Application {
                    op: cons,
                    args: vec![
                        Context::Hole,
                        Context::Hole,
                    ]
                }]
            },
            rhs: vec![Context::Hole],
        },
    ];
    let params = GeneticParams {
        max_sample_depth,
        n_crosses,
        p_add,
        p_keep,
        deterministic,
        templates,
        atom_weights
    };
    let mut task = make_task_from_data(&data[..0], otp.clone(), model_params);
    let s = Lexicon::from_signature(sig, ops, vars, knowledge);
    // FIXME: These shouldn't be constants.
    let h_star_lprior = 0.0;
    let h_star_llike = 0.0;
    let h_star_lpost = h_star_lprior + h_star_llike;

    let mut pop = s.init(&params, rng, &gp_params, &task);

    start_section("Evolving");
    for n_data in 0..=(data.len()) {
        if n_data > 0 {
            task = make_task_from_data(&data[0..n_data], otp.clone(), model_params);
            for i in pop.iter_mut() {
                i.1 = (task.oracle)(&s, &i.0);
            }
        }
        for gen in 0..generations_per_datum {
            s.evolve(&params, rng, &gp_params, &task, &mut pop);
            for (i, (individual, score)) in pop.iter().enumerate() {
                println!("{},{},{},{:.4},{:.4},{:?}",
                         n_data,
                         gen,
                         i,
                         score,
                         h_star_lpost - score,
                         individual.to_string(),
                );
            }
        }
    }

    start_section("Results");
    println!("rank,nlprior,nllike,nlpost,correct,better,difference");
    for (i, (individual, _)) in pop.iter().enumerate() {
        let nlprior = -individual.pseudo_log_prior();
        let nllike = -individual.log_likelihood(&data, model_params);
        let nlpost = nlprior + nllike;
        println!(
            "{},{:.4},{:.4},{:.4},{},{},{:.4},{:?}",
            i,
            nlprior,
            nllike,
            nlpost,
            (nllike <= h_star_llike) as usize,
            (nlpost <= h_star_lpost) as usize,
            h_star_lpost - nlpost,
            individual.to_string(),
        );
    }

    start_section("Summary");
    println!("best_score,best_difference,mean_score,mean_difference");
    let mean_score = pop.iter().map(|x| x.1).sum::<f64>()/(pop.len() as f64);
    println!("{:.4},{:.4},{:.4},{:.4}",
             pop[0].1,
             h_star_lpost - pop[0].1,
             mean_score,
             h_star_lpost - mean_score,
    );

    start_section("Done!");
    Ok(())
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
            let nil = Value::get_op(sig, ops, 0, Some("NIL"), &ptp!(list(tp!(int))));
            let cons = Value::get_op(sig, ops, 2, Some("CONS"), &ptp!(@arrow[tp!(int), tp!(list(tp!(int))), tp!(list(tp!(int)))]));
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
