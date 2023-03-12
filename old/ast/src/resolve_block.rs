use super::*;

impl Resolver {

    // resolve block when the return type is already known
    pub fn resolve_expected_block(&mut self,block: &Block,expected_type: &Type) -> Block {

        self.push_context(format!("block returning {}",expected_type));

        let mut stats: Vec<Stat> = Vec::new();
        for stat in block.stats.iter() {
            let new_stat = self.resolve_stat(stat);
            stats.push(new_stat);
        }
        let expr = if let Some(expr) = &block.expr {
            Some(Box::new(self.resolve_expected_expr(&expr,expected_type)))
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