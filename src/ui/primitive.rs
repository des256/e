/*
My take on this...

- node in a tree
- represents a rectangular area of screen
- can receive user events, sends them to Arc<> receivers
- can draw itself, either by using primitives, or by caching to texture, it possible/needed, like Flutter
- takes part in layouting, like Flutter

Is the Primitive going to be Rc<RefCell<>> hell?

*/

// attempt: make all primitives identical
pub struct Primitive {
    pub text: String,
    pub children: Vec<Primitive>,
}
/*
    pub fn update(&self,index: usize) {
        // rebuild slot
        self.children[index] = Rc::new(self.children[index].widget.realize(&Context { parent: &self,index: index, }));
        // TODO: relayout everything where needed
    }

    pub fn layout(&self,constraints: Constraints) {
    }

    pub fn render(&self,position: Position) {
    }
}*/
