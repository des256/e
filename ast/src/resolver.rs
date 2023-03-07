use super::*;

pub struct Resolver {
    pub module: Module,
    pub changes: Vec<String>,
    pub stdlib: StandardLib,
    pub context: Vec<String>,
}

impl Resolver {

    pub fn new(module: Module) -> Resolver {
        Resolver {
            module,
            changes: Vec::new(),
            stdlib: StandardLib::new(),
            context: Vec::new(),
        }
    }

    pub fn no_changes(&self) -> bool {
        self.changes.len() == 0
    }

    pub fn log_change(&mut self,change: String) {
        let mut r = String::new();
        let mut iter = self.context.iter();
        if let Some(context) = iter.next() {
            r += context;
            for context in iter {
                r += &format!(", {}",context);
            }
        }
        self.changes.push(format!("{}: {}",r,change));
    }

    pub fn push_context(&mut self,context: String) {
        self.context.push(context);
    }

    pub fn pop_context(&mut self) {
        self.context.pop();
    }
}

pub fn resolve(module: &Module) -> Module {
    let mut module = module.clone();
    println!("resolving module:");
    let mut cycle = 0;
    loop {
        let mut resolver = Resolver::new(module);
        module = resolver.resolve_module();
        if resolver.no_changes() {
            break;
        }
        println!("    cycle {}, {} changes:",cycle,resolver.changes.len());
        for change in resolver.changes.iter() {
            println!("        {}",change);
        }
        cycle += 1;
    }

    println!("result:\n{}",module);

    module
}
