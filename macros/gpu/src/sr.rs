// TODO: This needs to be the exact same file in src/gpu and in macros/shader/src

// Base Types
pub enum BaseType {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    Vec2U8,
    Vec2U16,
    Vec2U32,
    Vec2U64,
    Vec2I8,
    Vec2I16,
    Vec2I32,
    Vec2I64,
    Vec2F16,
    Vec2F32,
    Vec2F64,
    Vec3U8,
    Vec3U16,
    Vec3U32,
    Vec3U64,
    Vec3I8,
    Vec3I16,
    Vec3I32,
    Vec3I64,
    Vec3F16,
    Vec3F32,
    Vec3F64,
    Vec4U8,
    Vec4U16,
    Vec4U32,
    Vec4U64,
    Vec4I8,
    Vec4I16,
    Vec4I32,
    Vec4I64,
    Vec4F16,
    Vec4F32,
    Vec4F64,
    ColorU8,
    ColorU16,
    ColorF16,
    ColorF32,
    ColorF64,
}

#[allow(dead_code)]
pub fn base_type_size(base_type: &BaseType) -> usize {
    match base_type {
        BaseType::U8 => 1,
        BaseType::U16 => 2,
        BaseType::U32 => 4,
        BaseType::U64 => 8,
        BaseType::I8 => 1,
        BaseType::I16 => 2,
        BaseType::I32 => 4,
        BaseType::I64 => 8,
        BaseType::F16 => 2,
        BaseType::F32 => 4,
        BaseType::F64 => 8,
        BaseType::Vec2U8 => 2,
        BaseType::Vec2U16 => 4,
        BaseType::Vec2U32 => 8,
        BaseType::Vec2U64 => 16,
        BaseType::Vec2I8 => 2,
        BaseType::Vec2I16 => 4,
        BaseType::Vec2I32 => 8,
        BaseType::Vec2I64 => 16,
        BaseType::Vec2F16 => 4,
        BaseType::Vec2F32 => 8,
        BaseType::Vec2F64 => 16,
        BaseType::Vec3U8 => 3,
        BaseType::Vec3U16 => 6,
        BaseType::Vec3U32 => 12,
        BaseType::Vec3U64 => 24,
        BaseType::Vec3I8 => 3,
        BaseType::Vec3I16 => 6,
        BaseType::Vec3I32 => 12,
        BaseType::Vec3I64 => 24,
        BaseType::Vec3F16 => 6,
        BaseType::Vec3F32 => 12,
        BaseType::Vec3F64 => 24,
        BaseType::Vec4U8 => 4,
        BaseType::Vec4U16 => 8,
        BaseType::Vec4U32 => 16,
        BaseType::Vec4U64 => 32,
        BaseType::Vec4I8 => 4,
        BaseType::Vec4I16 => 8,
        BaseType::Vec4I32 => 16,
        BaseType::Vec4I64 => 32,
        BaseType::Vec4F16 => 8,
        BaseType::Vec4F32 => 16,
        BaseType::Vec4F64 => 32,
        BaseType::ColorU8 => 4,
        BaseType::ColorU16 => 8,
        BaseType::ColorF16 => 8,
        BaseType::ColorF32 => 16,
        BaseType::ColorF64 => 32,
    }
}

pub fn base_type_variant(base_type: &BaseType) -> &'static str {
    match base_type {
        BaseType::U8 => "U8",
        BaseType::U16 => "U16",
        BaseType::U32 => "U32",
        BaseType::U64 => "U64",
        BaseType::I8 => "I8",
        BaseType::I16 => "I16",
        BaseType::I32 => "I32",
        BaseType::I64 => "I64",
        BaseType::F16 => "F16",
        BaseType::F32 => "F32",
        BaseType::F64 => "F64",
        BaseType::Vec2U8 => "Vec2U8",
        BaseType::Vec2U16 => "Vec2U16",
        BaseType::Vec2U32 => "Vec2U32",
        BaseType::Vec2U64 => "Vec2U64",
        BaseType::Vec2I8 => "Vec2I8",
        BaseType::Vec2I16 => "Vec2I16",
        BaseType::Vec2I32 => "Vec2I32",
        BaseType::Vec2I64 => "Vec2I64",
        BaseType::Vec2F16 => "Vec2F16",
        BaseType::Vec2F32 => "Vec2F32",
        BaseType::Vec2F64 => "Vec2F64",
        BaseType::Vec3U8 => "Vec3U8",
        BaseType::Vec3U16 => "Vec3U16",
        BaseType::Vec3U32 => "Vec3U32",
        BaseType::Vec3U64 => "Vec3U64",
        BaseType::Vec3I8 => "Vec3I8",
        BaseType::Vec3I16 => "Vec3I16",
        BaseType::Vec3I32 => "Vec3I32",
        BaseType::Vec3I64 => "Vec3I64",
        BaseType::Vec3F16 => "Vec3F16",
        BaseType::Vec3F32 => "Vec3F32",
        BaseType::Vec3F64 => "Vec3F64",
        BaseType::Vec4U8 => "Vec4U8",
        BaseType::Vec4U16 => "Vec4U16",
        BaseType::Vec4U32 => "Vec4U32",
        BaseType::Vec4U64 => "Vec4U64",
        BaseType::Vec4I8 => "Vec4I8",
        BaseType::Vec4I16 => "Vec4I16",
        BaseType::Vec4I32 => "Vec4I32",
        BaseType::Vec4I64 => "Vec4I64",
        BaseType::Vec4F16 => "Vec4F16",
        BaseType::Vec4F32 => "Vec4F32",
        BaseType::Vec4F64 => "Vec4F64",
        BaseType::ColorU8 => "ColorU8",
        BaseType::ColorU16 => "ColorU16",
        BaseType::ColorF16 => "ColorF16",
        BaseType::ColorF32 => "ColorF32",
        BaseType::ColorF64 => "ColorF64",
    }
}

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
    Symbol(String),  // defined or built-in type `String`
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
