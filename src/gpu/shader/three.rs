struct Three {

}

enum Instruction {
    Block(Vec<(usize,Three)>),
    Loop(Vec<(usize,Three)>),
    If(usize,Vec<(usize,Three)>,Vec<(usize,Three)>),
    While(usize,Vec<(usize,Three)>),
}

struct Function {
    params: Vec<sr::BaseType>,
    return_type: Option<sr::BaseType>,
    instructions: Vec<Instruction>,
}

fn parse_function(item: sr::Item) -> Function {
    if let sr::Item::Function(_,sr_params,sr_return_ty,sr_stats) {
        let mut params
    }
    else {
        panic!("item should be function");
    }
}
