use crate::*;

pub struct PassContext {
    pub module: sr::Module,
    pub locals: HashMap<String,sr::Type>,
    pub params: HashMap<String,sr::Type>,
}

struct PassRun {
    pc: PassContext,
    handle_expr: fn(pc: &mut PassContext,expr: &sr::Expr) -> Option<sr::Expr>,
    handle_type: fn(pc: &mut PassContext,ty: &sr::Type) -> Option<sr::Type>,
    handle_pat: fn(pc: &mut PassContext,pat: &sr::Pat) -> Option<sr::Pat>,
}

impl PassRun {

    fn block(&mut self,block: &sr::Block) -> sr::Block {

        // save current local state
        let local_frame = self.locals.clone();

        // process each statement
        let mut new_stats: Vec<sr::Stat> = Vec::new();
        for stat in block.stats.iter() {
            new_stats.push(self.stat(stat));
        }

        // process the final expression
        let new_expr = if let Some(expr) = &block.expr {
            if let Some(expr) = self.handle_expr(&mut self.pc,expr) {
                Some(Box::new(expr))
            }
            else {
                Some(Box::new(self.expr(expr)))
            }
        }
        else {
            None
        };

        // restore local state
        self.locals = local_frame;

        // output new block
        sr::Block { stats: new_stats,expr: new_expr, }
    }

    pub fn expr(&mut self,expr: sr::Expr) -> Option<sr::Expr> {

    }

    pub fn ty(&mut self,ty: sr::Type) -> Option<sr::Type> {

    }

    pub fn pat(&mut self,pat: sr::Pat) -> Option<sr::Pat> {

    }
}

pub fn pass_module_expr(
    module: sr::Module,
    handle_expr: fn(pc: &mut PassContext,expr: &sr::Expr) -> Option<sr::Expr>,
    handle_type: fn(pc: &mut PassContext,ty: &sr::Type) -> Option<sr::Type>,
    handle_pat: fn(pc: &mut PassContext,pat: &sr::Pat) -> Option<sr::Pat>,
) -> sr::Module {

}
