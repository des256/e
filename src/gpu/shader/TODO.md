- sanitize gpu_macros and sr the same way
- we had to use RefCell constructions everywhere to make sure the Rc<>s still point to the right objects