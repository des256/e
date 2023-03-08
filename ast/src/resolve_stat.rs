use super::*;

impl Resolver {

    // resolve stat
    pub fn resolve_stat(&mut self,stat: &Stat) -> Stat {

        self.push_context(format!("{}",stat));

        self.pop_context();

        stat.clone()
    }
}
