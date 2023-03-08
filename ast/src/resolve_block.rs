use super::*;

impl Resolver {

    // resolve block when the return type is already known
    pub fn resolve_should_block(&mut self,block: &Block,should_type: &Type) -> Block {

        self.push_context(format!("block returning {}",should_type));

        let mut stats: Vec<Stat> = Vec::new();
        for stat in block.stats.iter() {
            let new_stat = self.resolve_stat(stat);
            stats.push(new_stat);
        }
        let expr = if let Some(expr) = &block.expr {
            Some(Box::new(self.resolve_should_expr(&expr,should_type)))
        }
        else {
            None
        };

        self.pop_context();

        Block {
            stats,
            expr,
        }
    }

    // resolve expr when the return type is not known
    pub fn resolve_block(&mut self,block: &Block) -> Block {

        self.push_context("block".to_string());

        let mut stats: Vec<Stat> = Vec::new();
        for stat in block.stats.iter() {
            let new_stat = self.resolve_stat(stat);
            stats.push(new_stat);
        }
        let expr = if let Some(expr) = &block.expr {
            Some(Box::new(self.resolve_expr(&expr)))
        }
        else {
            None
        };

        self.pop_context();

        Block {
            stats,
            expr,
        }
    }
}