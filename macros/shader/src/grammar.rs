// LEXICAL STUCTURE

// Vis = `pub` [ `(` `crate` | `self` | `super` | ( `in` SimplePath ) `)` ] .

// PathExpr = [ `::` ] PathExprSeg { `::` PathExprSeg }
pub(crate) struct PathExpr {
    pub abs: bool,
    pub segs: Vec<PathExprSeg>,
}

// PathExprSeg = PathIdentSeg [ Gens ]
pub(crate) struct PathExprSeg {
    pub seg: PathIdentSeg,
    pub gens: Option<Gens>,
}

// PathIdentSeg = IDENTIFIER | `super` | `self` | `Self` .
pub(crate) enum PathIdentSeg {
    Ident(String),
    Super,
    SelfInst,
    SelfType,
}

// Gens = `::` `<` [ Type { `,` Type } [ `,` ] ] [ GenAssign { `,` GenAssign } ] `>`
pub(crate) struct Gens {
    pub types: Vec<Box<Type>>,
    pub assigns: Vec<GenAssign>,
}

// GenAssign = IDENTIFIER `=` Type .
pub(crate) struct GenAssign {
    pub ident: String,
    pub ty: Box<Type>,
}

// PathType = [ `::` ] PathTypeSeg { `::` PathTypeSeg } .
pub(crate) struct PathType {
    pub abs: bool,
    pub segs: Vec<PathTypeSeg>,
}

// PathTypeSeg = PathIdentSeg [ PathTypeSegTail ] .
pub(crate) struct PathTypeSeg {
    pub seg: PathIdentSeg,
    pub tail: Option<PathTypeSegTail>,
}

// TypePathSegTail = Gens | TypePathFn .
pub(crate) enum PathTypeSegTail {
    Gen(Gens),
    Fn(PathTypeFn),
}

// PathTypeFn = `(` [ Type { `,` Type } [ `,` ] ] `)` [ `->` Type ] .
pub(crate) struct PathTypeFn {
    pub args: Vec<Box<Type>>,
    pub result: Option<Box<Type>>,
}

// QPathExpr = `<` Type [ `as` PathType ] `>` `::` PathExprSeg { `::` PathExprSeg } .
pub(crate) struct QPathExpr {
    pub ty: Box<Type>,
    pub as_ty: Option<PathType>,
    pub segs: Vec<PathExprSeg>,
}

// QPathType = `<` Type [ `as` PathType ] `>` `::` PathTypeSeg { `::` PathTypeSeg } .
pub(crate) struct QPathType {
    pub ty: Box<Type>,
    pub as_ty: Option<PathType>,
    pub segs: Vec<PathTypeSeg>,
}

// COMPILE ITEMS

// Item = [ Vis ] Func | Alias | Struct | Tuple | Enum | Union | Const | Static | Impl .
pub(crate) enum Item {
    Mod(String,Vec<Box<Item>>),  // `mod` IDENT `{` { Item } `}` .
    Func(String,Vec<FuncArg>,Option<Box<Type>>,Block),  // `fn` IDENT `(` [ FuncArg { `,` FuncArg } ] `)` [ `->` Type ] Block .
    Alias(String,Box<Type>),  // `type` IDENT `=` Type `;` .
    Struct(String,Vec<StructField>),  // `struct` IDENT ( `{` [ StructField { `,` StructField } [ `,` ] ] `}` ) | `;` .
    Tuple(String,Vec<Box<Type>>),  // `struct` IDENT `(` [ Type { `,` Type } [ `,` ] ] `)` `;` .
    Enum(String,Vec<EnumVar>),  // `enum` IDENT `{` [ EnumVar { `,` EnumVar } [ `,` ] ] `}` .
    Union(String,Vec<StructField>),  // `union` IDENT `{` [ StructField { `,` StructField } [ `,` ] ] `}` .
    Const(Option<String>,Box<Type>,Box<Expr>),  // `const` IDENT | `_` `:` Type `=` Expression `;` .
    Static(bool,String,Box<Type>,Box<Expr>),  // `static` [ `mut` ] IDENT `:` Type `=` Expression `;` .
    Impl(Box<Type>,Vec<ImplItem>),  // `impl` Type `{` { ImplItem } `}` .
}

// FuncArg = Pat `:` Type .
pub(crate) struct FuncArg {
    pub pat: Box<Pat>,
    pub ty: Box<Type>,
}

// StructField = [ Vis ] IDENT `:` Type .
pub(crate) struct StructField {
    pub ident: String,
    pub ty: Box<Type>,
}

// EnumVar = [ Vis ] IDENT [ EnumVarType ] .
pub(crate) struct EnumVar {
    pub ident: String,
    pub ty: Option<EnumVarType>,
}

// EnumVarType = ( `{` [ StructField { `,` StructField } [ `,` ] ] `}` ) | `(` [ Type { `,` Type } [ `,` ] ] `)` | Expr .
pub(crate) enum EnumVarType {
    Struct(Vec<StructField>),
    Tuple(Vec<Box<Type>>),
    Expr(Box<Expr>),
}

// ImplItem = [ Vis ] Const | Func | Method .
pub(crate) enum ImplItem {
    Const(Option<String>,Box<Type>,Box<Expr>),  // `const` IDENT | `_` `:` Type `=` Expr `;` .
    Func(String,Vec<FuncArg>,Option<Box<Type>>,Block),  // `fn` IDENT `(` [ FuncArg { `,` FuncArg } ] `)` [ `->` Type ] Block .
    Method(String,Option<SelfArg>,Vec<FuncArg>,Option<Box<Type>>,Block),  // `fn` IDENT `(` ( SelfArg { `,` FuncArg } ) | ( FuncArg { `,` FuncArg } ) [ `,` ] ) [ `->` Type ] Block .
}

// SelfArg = Shorthand | Typed .
pub(crate) enum SelfArg {
    Once,  // self
    Ref,  // &self
    MutRef,  // &mut self
    Typed(Box<Type>),  // self: Type
    RefTyped(Box<Type>),  // self: &Type
    MutTyped(Box<Type>),  // self: &mut Type
}

// STATEMENTS AND EXPRESSIONS

// Stat = `;` | Item | LetStat | ExprStat .
pub(crate) enum Stat {
    Empty,
    Item(Box<Item>),
    Let(LetStat),
    Expr(Box<Expr>),
}

// LetStat = `let` Pat [ `:` Type ] [ `=` Expr ] `;` .
pub(crate) struct LetStat {
    pub pat: Box<Pat>,
    pub ty: Option<Box<Type>>,
    pub expr: Option<Box<Expr>>,
}

pub(crate) enum Expr {
    Char(char),
    String(String),
    RawString(String),
    Byte(u8),
    ByteString(Vec<u8>),
    RawByteString(Vec<u8>),
    Int(isize),
    Float(f64),
    Bool(bool),
    Path(PathExpr),
    QPath(QPathExpr),
    Borrow(bool,bool,Box<Expr>),  // `&` | `&&` [ `mut` ] Expr .
    Deref(Box<Expr>), // `*` Expr .
    Error(Box<Expr>), // Expr `?` .
    Unary(UnaryOp,Box<Expr>),  // UnaryOp Expr .
    Binary(Box<Expr>,BinaryOp,Box<Expr>),  // Expr BinaryOp Expr .
    Comp(Box<Expr>,CompOp,Box<Expr>),  // Expr CompOp Expr .
    Cast(Box<Expr>,Box<Type>),  // Expr `as` Type .
    Assign(Box<Expr>,Box<Expr>),  // Expr `=` Expr .
    CompAssign(Box<Expr>,CompAssignOp,Box<Expr>),  // Expr CompAssignOp Expr .
    Grouped(Box<Expr>), // `(` Expr `)` .
    Array(ArrayElements),
    Await(Box<Expr>), // Expr `.` `await` .
    Index(Box<Expr>,Box<Expr>),  // Expr `[` Expr `]` .
    TupleIndex(Box<Expr>,usize),  // Expr `.` usize .
    Tuple(Vec<Box<Expr>>), // `(` [ Expr { `,` Expr } [ `,` ] ] `)` .
    Struct(StructExprStruct),
    TupleStruct(StructExprTuple),
    UnitStruct(PathExpr),
    Enum(EnumExpr),
    Call(Box<Expr>,Vec<Box<Expr>>),  // Expr `(` [ Expr { `,` Expr } [ `,` ] ] `)` .
    Method(Box<Expr>,PathExprSeg,Vec<Box<Expr>>),  // Expr `.` PathExprSeg `(` [ Expr { `,` Expr } [ `,` ] ] `)` .
    Field(Box<Expr>,String),  // Expr `.` IDENT .
    Continue,  // `continue` .
    Break(Option<Box<Expr>>),  // `break` [ Expr ] .
    RangeFromToIncl(Box<Expr>,Box<Expr>),  // Expr `..=` Expr .
    RangeFromTo(Box<Expr>,Box<Expr>),  // Expr `..` Expr .
    RangeFrom(Box<Expr>),  // Expr `..` .
    RangeToIncl(Box<Expr>),  // `..=` Expr .
    RangeTo(Box<Expr>),  // `..` Expr .
    RangeAll,  // `..` .
    Return(Box<Expr>),  // `return` [ Expr ] .
    Block(Block),
    Async(bool,Block),  // `async` [ `move` ] Block .
    Unsafe(Block),  // `unsafe` Block .
    Loop(Block),  // `loop` Block .
    While(Box<Expr>,Block),  // `while` Expr Block .
    WhileLet(Box<Pat>,Box<Expr>,Block),  // `while` `let` MatchArmsPats `=` Expr Block .
    For(Box<Pat>,Box<Expr>,Block),  // `for` Pat `in` Expr Block .
    If(IfExpr),  
    IfLet(IfLetExpr),
    Match(Box<Expr>,Vec<MatchArmExpr>),  // `match` Expr `{` [ MatchArmExpr { `,` MatchArmExpr } [ `,` ] ] `}` .
}

// Block = `{` { Stat } [ ExprWithoutBlock ] `}` .
pub(crate) struct Block {
    pub exprs: Vec<Box<Expr>>,
    pub result: Option<Box<Expr>>,
}

// UnaryOp = `-` | `!` .
pub(crate) enum UnaryOp {
    Minus,
    Not,
}

// BinaryOp = `+` | `-` | `*` | `/` | `%` | `&` | `|` | `^` | `<<` | `>>` .
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    LAnd,
    LOr,
}

// CompOp = `==` | `!=` | `<` | `>` | `<=` | `>=` .
pub(crate) enum CompOp {
    Equal,
    NotEqual,
    Greater,
    Less,
    NotGreater,
    NotLess,
}

// CompAssignOp = `+=` | `-=` | `*=` | `/=` | `%=` | `&=` | `|=` | `^=` | `<<=` | `>>=` .
pub(crate) enum CompAssignOp {
    AddEq,
    SubEq,
    MulEq,
    DivEq,
    ModEq,
    AndEq,
    OrEq,
    XorEq,
    ShlEq,
    ShrEq,
}

// ArrayElements = ( Expr { `,` Expr } [ `,` ] ) | ArrayRange .
pub(crate) enum ArrayElements {
    Single(Vec<Box<Expr>>),
    Range(Box<Expr>,Box<Expr>),  // Expr `;` Expr .
}

// StructExprStruct = PathExpr `{` [ StructExprField { `,` StructExprField } ] [ `..` Expr ] [ `,` ] `}` .
pub(crate) struct StructExprStruct {
    pub expr: PathExpr,
    pub fields: Vec<StructExprField>,
    pub base: Option<Box<Expr>>,
}

// IdentOrIndex = IDENT | INDEX .
pub(crate) enum IdentOrIndex {
    Ident(String),
    Index(usize),
}

// StructExprField = IdentOrIndex [ `:` Expr ] .
pub(crate) struct StructExprField {
    pub ident_or_index: IdentOrIndex,
    pub expr: Option<Box<Expr>>,
}

// StructExprTuple = PathExpr `(` [ Expr { `,` Expr } [ `,` ] ] `)` .
pub(crate) struct StructExprTuple {
    pub expr: PathExpr,
    pub parts: Vec<Box<Expr>>,
}

// EnumExpr = EnumExprStruct | EnumExprTuple | EnumExprFieldless .
pub(crate) enum EnumExpr {
    Struct(PathExpr,Vec<EnumExprField>),  // PathExpr `{` [ EnumExprField { `,` EnumExprField } [ `,` ] ] `}` .
    Tuple(PathExpr,Vec<Box<Expr>>),  // PathExpr `(` [ Expr { `,` Expr } [ `,` ] ] `)` .
    Fieldless(PathExpr),
}

// EnumExprField = IDENTIFIER | ( IDENTIFIER | TUPLE_INDEX `:` Expression ) .
pub(crate) enum EnumExprField {
    Ident(String),
    IdentExpr(String,Box<Expr>),
    IndexExpr(usize,Box<Expr>),
}

// ElseExpr = Block | IfExpr | IfLetExpr .
pub(crate) enum ElseExpr {
    Block(Block),
    If(IfExpr),
    IfLet(IfLetExpr),
}

// `if` Expr Block [ `else` ElseExpr ] .
pub(crate) struct IfExpr {
    pub cond: Box<Expr>,
    pub block: Block,
    pub el: Option<Box<ElseExpr>>,
}

// IfLetExpr = `if` `let` MatchArmPats `=` StructlessLazylessExpr Block [ `else` ElseExpr ] .
pub(crate) struct IfLetExpr {
    pub pats: Vec<Pat>,
    pub expr: Box<Expr>,
    pub block: Block,
    pub el: Box<ElseExpr>,
}

// MatchArmExpr = [ `|` ] Pat { `|` Pat } [ `if` Expr ] `=>` Expr .
pub(crate) struct MatchArmExpr {
    pub pats: Vec<Box<Pat>>,
    pub guard: Option<Box<Expr>>,
    pub expr: Box<Expr>,
}

// PATTERNS

// Pat = Literal | Ident | Wildcard | Rest | ObsRange | Ref | Struct | TupleStruct | Tuple | Grouped | Slice | Path | QPath | Range .
pub(crate) enum Pat {
    Bool(bool),
    Char(char),
    Byte(u8),
    String(String),
    RawString(String),
    ByteString(Vec<u8>),
    RawByteString(Vec<u8>),
    Integer(isize),
    Float(u64),
    Ident(bool,bool,String,Option<Box<Pat>>),  // [ `ref` ] [ `mut` ] String [ `@` Pat ] .
    Wildcard,  // `_` .
    Rest,  // `..` .
    Range(RangePatBound,RangePatBound),  // RangePatBound `..=` RangePatBound .
    ObsRange(RangePatBound,RangePatBound),  // RangePatBound `...` RangePatBound .
    Ref(bool,bool,Box<Pat>),  // `&` | `&&` [ `mut` ] Pat .
    Struct(PathExpr,Vec<StructPatField>,bool),  // PathExpr `{` [ StructPatField { `,` StructPatField } ] [ `,` `..` ] `}` .
    TupleStruct(PathExpr,Vec<Box<Pat>>),  // PathExpr `(` [ Pat { `,` Pat } [ `,` ] ] `)` .
    Tuple(Vec<Box<Pat>>),  // `(` [ Pat { `,` Pat } [ `,` ] ] `)` .
    Grouped(Box<Pat>), // `(` Pat `)` .
    Slice(Vec<Box<Pat>>),  // `[` [ Pat { `,` Pat } [ `,` ] ] `]` .
    Path(PathExpr),
    QPath(QPathExpr),
}

// RangePatBound = CHAR | BYTE | Path | QPath | INTEGER | FLOAT .
pub(crate) enum RangePatBound {
    Char(char),
    Byte(u8),
    Path(PathExpr),
    QPath(QPathExpr),
    Integer(isize),
    Float(f64),
}

// StructPatField = SPFTuple | SPFStruct | SPFIdent .
pub(crate) enum StructPatField {
    Tuple(usize,Box<Pat>),  // usize `:` Pat .
    Struct(String,Box<Pat>),  // String `:` Pat .
    Ident(bool,bool,String),  // [ `ref` ] [ `mut` ] IDENT .
}

// TYPE SYSTEM

// Type = Paren | Path | Tuple | Never | RawPointer | Ref | Array | Slice | Inferred | QPath | BareFunc .
pub(crate) enum Type {
    Paren(Box<Type>),  // `(` Type `)` .
    Path(PathType),
    Tuple(Vec<Box<Type>>), // `(` [ Type { , Type } [ `,` ] ] `)` .
    Never, // `!` .
    RawPointer(bool,Box<Type>),  // `*` `mut` | `const` Type .
    Ref(bool,Box<Type>),  // [ `mut` ] Type .
    Array(Box<Type>,Box<Expr>),  // `[` Type `;` Expr `]` .
    Slice(Box<Type>), // `[` Type `]` .
    Inferred, // `_` .
    QPath(QPathType),
    BareFunc(BareFuncType),
}

// BareFuncType = `fn` `(` [ MaybeNamedParam { `,` MaybeNamedParam } [ `,` ] [ `...` ] ] `)` [ `->` Type ] .
pub(crate) struct BareFuncType {
    pub params: Vec<MaybeNamedParam>,
    pub is_variadic: bool,
    pub result: Option<Box<Type>>,
}

// MaybeNamedParam = [ IDENT | `_` `:` ] Type .
pub(crate) struct MaybeNamedParam {
    pub ident: Option<Option<String>>,
    pub ty: Box<Type>,
}
