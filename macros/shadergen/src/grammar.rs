// LEXICAL STUCTURE

// Vis = `pub` [ `(` `crate` | `self` | `super` | ( `in` SimplePath ) `)` ] .

// PathExpr = [ `::` ] PathExprSeg { `::` PathExprSeg }
struct PathExpr {
    absolute: bool,
    segments: Vec<PathExprSeg>,
}

// PathExprSeg = PathIdentSeg [ Gens ]
struct PathExprSeg {
    segment: PathIdentSeg,
    generics: Option<Gens>,
}

// PathIdentSeg = IDENTIFIER | `super` | `self` | `Self` .
enum PathIdentSeg {
    Ident(String),
    Super,
    SelfInst,
    SelfType,
}

// Gens = `::` `<` [ Type { `,` Type } [ `,` ] ] [ GenAssign { `,` GenAssign } ] `>`
struct Gens {
    types: Vec<Type>,
    assigns: Vec<GenAssign>,
}

// GenAssign = IDENTIFIER `=` Type .
struct GenAssign {
    ident: String,
    ty: Type,
}

// PathType = [ `::` ] PathTypeSeg { `::` PathTypeSeg } .
struct PathType {
    absolute: bool,
    segments: Vec<PathTypeSeg>,
}

// PathTypeSeg = PathIdentSeg [ PathTypeSegTail ] .
struct PathTypeSeg {
    segment: PathIdentSeg,
    tail: Option<PathTypeSegTail>,
}

// TypePathSegTail = Gens | TypePathFn .
enum PathTypeSegTail {
    Gen(Gens),
    Fn(PathTypeFn),
}

// PathTypeFn = `(` [ Type { `,` Type } [ `,` ] ] `)` [ `->` Type ] .
struct PathTypeFn {
    args: Vec<Type>,
    result: Option<Type>,
}

// QPathExpr = `<` Type [ `as` PathType ] `>` `::` PathExprSeg { `::` PathExprSeg } .
struct QPathExpr {
    ty: Type,
    as_ty: Option<PathType>,
    segments: Vec<PathExprSeg>,
}

// QPathType = `<` Type [ `as` PathType ] `>` `::` PathTypeSeg { `::` PathTypeSeg } .
struct QPathType {
    ty: Type,
    as_ty: Option<PathType>,
    segments: Vec<PathTypeSeg>,
}

// COMPILE ITEMS

// Decl = [ Vis ] Func | Alias | Struct | Tuple | Enum | Union | Const | Static | Impl .
enum Decl {
    Func(Func),
    Alias(Alias),
    Struct(Struct),
    Tuple(Tuple),
    Enum(Enum),
    Union(Union),
    Const(Const),
    Static(Static),
    Impl(Impl),
}

// AsyncConst = `async` | `const` .
enum AsyncConst {
    Async,
    Const,
}

// Func = [ AsyncConst ] `fn` IDENT `(` [ FuncArg { `,` FuncArg } ] `)` [ `->` Type ] Block .
struct Func {
    async_const: Option<AsyncConst>,
    ident: String,
    args: Vec<FuncArg>,
    result: Option<Type>,
    code: Block,
}

// FuncArg = Pat `:` Type .
struct FuncArg {
    pat: Pat,
    ty: Type,
}

// Alias = `type` IDENT `=` Type `;` .
struct Alias {
    ident: String,
    ty: Type,
}

// Struct = `struct` IDENT ( `{` [ StructField { `,` StructField } [ `,` ] ] `}` ) | `;` .
struct Struct {
    ident: String,
    fields: Vec<StructField>,
}

// StructField = [ Vis ] IDENT `:` Type .
struct StructField {
    ident: String,
    ty: Type,
}

// Tuple = `struct` IDENT `(` [ Type { `,` Type } [ `,` ] ] `)` `;` .
struct Tuple {
    ident: String,
    fields: Vec<Type>,
}

// Enum = `enum` IDENT `{` [ EnumVar { `,` EnumVar } [ `,` ] ] `}` .
struct Enum {
    ident: String,
    vars: Vec<EnumVar>,
}

// EnumVar = [ Vis ] IDENT [ EnumVarType ] .
struct EnumVar {
    ident: String,
    ty: Option<EnumVarType>,
}

// EnumVarType = ( `{` [ StructField { `,` StructField } [ `,` ] ] `}` ) | `(` [ Type { `,` Type } [ `,` ] ] `)` | Expr .
enum EnumVarType {
    Struct(Vec<StructField>),
    Tuple(Vec<Type>),
    Expr(Expr),
}

// Union = `union` IDENT `{` [ StructField { `,` StructField } [ `,` ] ] `}` .
struct Union {
    ident: String,
    fields: Vec<StructField>,
}

// Const = `const` IDENT | `_` `:` Type `=` Expression `;` .
struct Const {
    ident: Option<String>,
    ty: Type,
    expr: Expr,
}

// Static = `static` [ `mut` ] IDENT `:` Type `=` Expression `;` .
struct Static {
    is_mut: bool,
    ident: String,
    ty: Type,
    expr: Expr,
}

// Impl = `impl` Type `{` { ImplItem } `}` .
struct Impl {
    ty: Type,
    items: Vec<ImplItem>,
}

// ImplItem = [ Vis ] Const | Func | Method .
enum ImplItem {
    Const(Const),
    Func(Func),
    Method(Method),
}

// Method = [ AsyncConst ] `fn` IDENT `(` ( SelfArg { `,` FuncArg } ) | ( FuncArg { `,` FuncArg } ) [ `,` ] ) [ `->` Type ] Block .
struct Method {
    async_const: Option<AsyncConst>,
    ident: String,
    self_arg: Option<SelfArg>,
    args: Vec<FuncArg>,
    result: Option<Type>,
    code: Block,
}

// SelfArg = Shorthand | Typed .
enum SelfArg {
    Once,  // self
    Ref,  // &self
    MutRef,  // &mut self
    Typed(Type),  // self: Type
    RefTyped(Type),  // self: &Type
    MutTyped(Type),  // self: &mut Type
}

// STATEMENTS AND EXPRESSIONS

// Stat = `;` | Decl | LetStat | ExprStat .
enum Stat {
    Empty,
    Decl(Decl),
    Let(LetStat),
    Expr(Expr),
}

// LetStat = `let` Pat [ `:` Type ] [ `=` Expr ] `;` .
struct LetStat {
    pat: Pat,
    ty: Option<Type>,
    expr: Option<Expr>,
}

enum Expr {
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
    Borrow(bool,bool,Expr),  // `&` | `&&` [ `mut` ] Expr .
    Deref(Expr), // `*` Expr .
    Error(Expr), // Expr `?` .
    Unary(UnaryOp,Expr),  // UnaryOp Expr .
    Binary(Expr,BinaryOp,Expr),  // Expr BinaryOp Expr .
    Comp(Expr,CompOp,Expr),  // Expr CompOp Expr .
    Cast(Expr,Type),  // Expr `as` Type .
    Assign(Expr,Expr),  // Expr `=` Expr .
    CompAssign(Expr,CompAssignOp,Expr),  // Expr CompAssignOp Expr .
    Grouped(Expr), // `(` Expr `)` .
    Array(ArrayElements),
    Await(Expr), // Expr `.` `await` .
    Index(Expr,Expr),  // Expr `[` Expr `]` .
    TIndex(TIndexExpr),  // Expr `.` usize .
    Tuple(Vec<Expr>), // `(` [ Expr { `,` Expr } [ `,` ] ] `)` .
    Struct(StructExprStruct),
    TupleStruct(StructExprTuple),
    UnitStruct(PathExpr),
    Enum(EnumExpr),
    Call(Expr,Vec<Expr>),  // Expr `(` [ Expr { `,` Expr } [ `,` ] ] `)` .
    Method(Expr,PathExprSeg,Vec<Expr>),  // Expr `.` PathExprSeg `(` [ Expr { `,` Expr } [ `,` ] ] `)` .
    Field(Expr,String),  // Expr `.` IDENT .
    Continue,  // `continue` .
    Break(Option<Expr>),  // `break` [ Expr ] .
    RangeFromToIncl(Expr,Expr),  // Expr `..=` Expr .
    RangeFromTo(Expr,Expr),  // Expr `..` Expr .
    RangeFrom(Expr),  // Expr `..` .
    RangeToIncl(Expr),  // `..=` Expr .
    RangeTo(Expr),  // `..` Expr .
    RangeAll,  // `..` .
    Return(Expr),  // `return` [ Expr ] .
    Block(Block),
    Async(bool,Block),  // `async` [ `move` ] Block .
    Unsafe(Block),  // `unsafe` Block .
    Loop(Block),  // `loop` Block .
    While(Expr,Block),  // `while` Expr Block .
    WhileLet(MatchArmsPats,Expr,Block),  // `while` `let` MatchArmsPats `=` Expr Block .
    For(Pat,Expr,Block),  // `for` Pat `in` Expr Block .
    If(IfExpr),  
    IfLet(IfLetExpr),
    Match(Expr,Vec<MatchArmExpr>),  // `match` Expr `{` [ MatchArmExpr { `,` MatchArmExpr } [ `,` ] ] `}` .
}

// Block = `{` { Stat } [ ExprWithoutBlock ] `}` .
struct Block {
    exprs: Vec<Expr>,
    result: Option<Expr>,
}

// UnaryOp = `-` | `!` .
enum UnaryOp {
    Minus,
    Not,
}

// BinaryOp = `+` | `-` | `*` | `/` | `%` | `&` | `|` | `^` | `<<` | `>>` .
enum BinaryOp {
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
enum CompOp {
    Equal,
    NotEqual,
    Greater,
    Less,
    NotGreater,
    NotLess,
}

// CompAssignOp = `+=` | `-=` | `*=` | `/=` | `%=` | `&=` | `|=` | `^=` | `<<=` | `>>=` .
enum CompAssignOp {
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
enum ArrayElements {
    Single(Vec<Expr>),
    Range(Expr,Expr),  // Expr `;` Expr .
}

// StructExprStruct = PathExpr `{` [ StructExprField { `,` StructExprField } ] [ `..` Expr ] [ `,` ] `}` .
struct StructExprStruct {
    expr: PathExpr,
    fields: Vec<StructExprField>,
    base: Option<Expr>,
}

// IdentOrIndex = IDENT | INDEX .
enum IdentOrIndex {
    Ident(String),
    Index(usize),
}

// StructExprField = IdentOrIndex [ `:` Expr ] .
struct StructExprField {
    ident_or_index: IdentOrIndex,
    expr: Option<Expr>,
}

// StructExprTuple = PathExpr `(` [ Expr { `,` Expr } [ `,` ] ] `)` .
struct StructExprTuple {
    expr: PathExpr,
    parts: Vec<Expr>,
}

// EnumExpr = EnumExprStruct | EnumExprTuple | EnumExprFieldless .
enum EnumExpr {
    Struct(PathExpr,Vec<EnumExprField>),  // PathExpr `{` [ EnumExprField { `,` EnumExprField } [ `,` ] ] `}` .
    Tuple(PathExpr,Vec<Expr>),  // PathExpr `(` [ Expr { `,` Expr } [ `,` ] ] `)` .
    Fieldless(PathExpr),
}

// ElseExpr = Block | IfExpr | IfLetExpr .
enum ElseExpr {
    Block(Block),
    If(IfExpr),
    IfLet(IfLetExpr),
}

// `if` Expr Block [ `else` ElseExpr ] .
struct IfExpr {
    cond: Expr,
    block: Block,
    el: Option<ElseExpr>,
}

// IfLetExpr = `if` `let` MatchArmPats `=` StructlessLazylessExpr Block [ `else` ElseExpr ] .
struct IfLetExpr {
    pats: Vec<Pat>,
    expr: Expr,
    block: Block,
    el: ElseExpr,
}

// MatchArmExpr = [ `|` ] Pat { `|` Pat } [ `if` Expr ] `=>` Expr .
struct MatchArmExpr {
    pats: Vec<Pattern>,
    guard: Option<Expr>,
    expr: Expr,
}

// PATTERNS

// Pat = Literal | Ident | Wildcard | Rest | ObsRange | Ref | Struct | TupleStruct | Tuple | Grouped | Slice | Path | QPath | Range .
enum Pat {
    Bool(bool),
    Char(char),
    Byte(u8),
    String(String),
    RawString(String),
    ByteString(Vec<u8>),
    RawByteString(Vec<u8>),
    Integer(isize),
    Float(u64),
    Ident(bool,bool,String,Option<Pat>),  // [ `ref` ] [ `mut` ] String [ `@` Pat ] .
    Wildcard,  // `_` .
    Rest,  // `..` .
    Range(RangePatBound,RangePatBound),  // RangePatBound `..=` RangePatBound .
    ObsRange(RangePatBound,RangePatBound),  // RangePatBound `...` RangePatBound .
    Ref(bool,bool,Pat),  // `&` | `&&` [ `mut` ] Pat .
    Struct(PathExpr,Vec<StructPatField>,bool),  // PathExpr `{` [ StructPatField { `,` StructPatField } ] [ `,` `..` ] `}` .
    TupleStruct(PathExpr,Vec<Pat>),  // PathExpr `(` [ Pat { `,` Pat } [ `,` ] ] `)` .
    Tuple(Vec<Pat>),  // `(` [ Pat { `,` Pat } [ `,` ] ] `)` .
    Grouped(Pat), // `(` Pat `)` .
    Slice(Vec<Pat>),  // `[` [ Pat { `,` Pat } [ `,` ] ] `]` .
    Path(PathExpr),
    QPath(QPathExpr),
}

// RangePatBound = CHAR | BYTE | Path | QPath | INTEGER | FLOAT .
enum RangePatBound {
    Char(char),
    Byte(u8),
    Path(PathExpr),
    QPath(QPathExpr),
    Integer(isize),
    Float(f64),
}

// StructPatField = SPFTuple | SPFStruct | SPFIdent .
enum StructPatField {
    Tuple(usize,Pat),  // usize `:` Pat .
    Struct(String,Pat),  // String `:` Pat .
    Ident(bool,bool,String),  // [ `ref` ] [ `mut` ] IDENT .
}

// TYPE SYSTEM

// Type = Paren | Path | Tuple | Never | RawPointer | Ref | Array | Slice | Inferred | QPath | BareFunc .
enum Type {
    Paren(Type),  // `(` Type `)` .
    Path(PathType),
    Tuple(Vec<Type>), // `(` [ Type { , Type } [ `,` ] ] `)` .
    Never, // `!` .
    RawPointer(bool,Type),  // `*` `mut` | `const` Type .
    Ref(RefType),  // [ `mut` ] Type .
    Array(ArrayType),  // `[` Type `;` Expr `]` .
    Slice(Type), // `[` Type `]` .
    Inferred, // `_` .
    QPath(QPathType),
    BareFunc(BareFuncType),
}

// BareFuncType = [ AsyncConst ] `fn` `(` [ MaybeNamedParam { `,` MaybeNamedParam } [ `,` ] [ `...` ] ] `)` [ `->` Type ] .
struct BareFuncType {
    async_const: Option<AsyncConst>,
    params: Vec<MaybeNamedParam>,
    is_variadic: bool,
    result: Option<Type>,
}

// MaybeNamedParam = [ IDENT | `_` `:` ] Type .
struct MaybeNamedParam {
    ident: Option<Option<String>>,
    ty: Type,
}
