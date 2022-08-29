use crate::*;

// Literals
#[allow(dead_code)]
pub enum Literal {
    Bool(bool),  // `true` or `false`
    Char(char),  // `'C'`
    String(String),  // `"STRING"`
    Integer(u64),  // integer constant
    Float(f64),  // floating point constant
}

// Patterns
pub enum Pat {
    Wildcard,  // matching anything `_`
    Rest,  // matching the rest `..`
    Literal(String),  // matching a specific literal
    Slice(Vec<Pat>),  // matching a slice of patterns `[Pat,...,Pat]`
    Symbol(String),  // matching a named variable or constant `String`
}

// Types
pub enum Type {
    Array(Box<Type>,Box<Expr>),  // array specification `Type[Expr]`
    Tuple(Vec<Type>),  // tuple specification `(Type,...,Type)`
    Symbol(String),  // reference to external type
    BaseType(BaseType),  // built-in type
    Inferred,  // inferred type `_`
}

// Expressions
pub enum Expr {
    Literal(String),  // literal
    Symbol(String),  // named variable or constant
    AnonArray(Vec<Expr>),  // anonymous array literal `[Expr,...,Expr]`
    AnonTuple(Vec<Expr>),  // anonymous tuple literal `(Expr,...,Expr)`
    AnonCloned(Box<Expr>,Box<Expr>),  // anonymous cloned array literal `[Expr; Expr]`
    Struct(String,Vec<(String,Expr)>),  // struct literal `String { String: Expr,...,String: Expr }`
    Tuple(String,Vec<Expr>),  // tuple literal `String ( Expr,...,Expr )`
    Field(Box<Expr>,String),  // a field of a structure `Expr.field`
    Index(Box<Expr>,Box<Expr>),  // one element from an array `Expr[Expr]`
    Call(Box<Expr>,Vec<Expr>),  // function call `Expr(Expr,...,Expr)`
    Error(Box<Expr>),  // error propagation `Expr?`
    Cast(Box<Expr>,Box<Type>),  // cast to other type `Expr as Type`
    Neg(Box<Expr>),  // negate `-Expr`
    Not(Box<Expr>),  // bitwise or logical not `!Expr`
    Mul(Box<Expr>,Box<Expr>),  // multiplication `Expr * Expr`
    Div(Box<Expr>,Box<Expr>),  // division `Expr / Expr`
    Mod(Box<Expr>,Box<Expr>),  // modulus `Expr % Expr`
    Add(Box<Expr>,Box<Expr>),  // addition `Expr + Expr`
    Sub(Box<Expr>,Box<Expr>),  // subtraction `Expr - Expr`
    Shl(Box<Expr>,Box<Expr>),  // bitwise shift left `Expr << Expr`
    Shr(Box<Expr>,Box<Expr>),  // bitwise shift right `Expr >> Expr`
    And(Box<Expr>,Box<Expr>),  // bitwise and `Expr & Expr`
    Xor(Box<Expr>,Box<Expr>),  // bitwise xor `Expr ^ Expr`
    Or(Box<Expr>,Box<Expr>),   // bitwise or `Expr | Expr`
    Eq(Box<Expr>,Box<Expr>),  // is equal to `Expr == Expr`
    NotEq(Box<Expr>,Box<Expr>),  // is not equal to `Expr != Expr`
    Gt(Box<Expr>,Box<Expr>),  // is greater than `Expr > Expr`
    NotGt(Box<Expr>,Box<Expr>),  // is less than or equal to `Expr <= Expr`
    Lt(Box<Expr>,Box<Expr>),  // is less than `Expr < Expr`
    NotLt(Box<Expr>,Box<Expr>),  // is greater than or equal to `Expr >= Expr`
    LogAnd(Box<Expr>,Box<Expr>),  // logical and `Expr && Expr`
    LogOr(Box<Expr>,Box<Expr>),  // logical or `Expr || Expr`
    Assign(Box<Expr>,Box<Expr>),  // assignment `Expr = Expr`
    AddAssign(Box<Expr>,Box<Expr>),  // add-assignment `Expr += Expr`
    SubAssign(Box<Expr>,Box<Expr>),  // subtract-assignment `Expr -= Expr`
    MulAssign(Box<Expr>,Box<Expr>),  // multiply-assignment `Expr *= Expr`
    DivAssign(Box<Expr>,Box<Expr>),  // divide-assignment `Expr /= Expr`
    ModAssign(Box<Expr>,Box<Expr>),  // modulo-assignment `Expr %= Expr`
    AndAssign(Box<Expr>,Box<Expr>),  // bitwise and-assignment `Expr &= Expr`
    XorAssign(Box<Expr>,Box<Expr>),  // bitwise xor-assignment `Expr ^= Expr`
    OrAssign(Box<Expr>,Box<Expr>),  // bitwise or-assignment `Expr |= Expr`
    Block(Vec<Stat>),  //  block expression `{ Stat...Stat }`
    Continue,  // continue next iteration `continue`
    Break(Option<Box<Expr>>),  // break loop `break Expr`
    Return(Option<Box<Expr>>),  // return from function `return Expr`
    Loop(Vec<Stat>), // infinite loop `loop { Stat...Stat }`
    For(Pat,Box<Expr>,Vec<Stat>),  // for-loop `for Pat = Expr { Stat ... Stat }`
    If(Box<Expr>,Vec<Stat>,Option<Box<Expr>>),  // if expression `if Expr { Stat ... Stat } else Expr`
    IfLet(Vec<Pat>,Box<Expr>,Vec<Stat>,Option<Box<Expr>>),  // predicated if expression `if let Pat | ... | Pat = Expr { Stat ... Stat } else Expr`
    While(Box<Expr>,Vec<Stat>),  // while loop `while Expr { Stat ... Stat }`
    WhileLet(Vec<Pat>,Box<Expr>,Vec<Stat>),  // predicated while loop `while let Pat | ... | Pat = Expr { Stat ... Stat }`
    Match(Box<Expr>,Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)>),  // match expression `match Expr { Pat | ... | Pat if Expr => Expr, ..., Pat | ... | Pat if Expr => Expr }`
}

// Statements
pub enum Stat {
    Expr(Box<Expr>),  // statement expression `Expr;`
    Let(Pat,Option<Box<Type>>,Option<Box<Expr>>),  // declaration `let Pat: Type = Expr;`
}

// Items
pub enum Item {
    Module(String,Vec<Item>),  // module definition `mod String { Item ... Item }`
    Function(String,Vec<(Pat,Box<Type>)>,Option<Box<Type>>,Vec<Stat>),  // function definition `fn String(Pat: Type,...,Pat: Type) -> Type { Stat ... Stat }`
    Struct(String,Vec<(String,Box<Type>)>),  // struct definition `struct String { String: Type,...,String: Type }`
}
