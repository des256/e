# Reduced Rust Grammar

Ultimately we want to describe shaders in Rust, so this grammar is without:

- macros
- module hierarchy
- use declarations
- lifetimes
- traits
- closures
- where clauses
- methods and implementations
- async/await
- labels

And ignore:

- InnerAttribute = `#` `!` `[` Attr `]` .
- OuterAttribute = `#` `[` Attr `]` .
- Visibility = `pub` [ `(` `crate` | `self` | `super` | ( `in` SimplePath ) `)` ] .
- unsafe keyword

Also, the Rust compiler itself is compiling the shader too, so this parser will not have to fix errors, only generate a parse tree.

## Lexical Structure

TypeList = Type { `,` Type } [ `,` ] .

PathTypeFn = `(` [ TypeList ] `)` [ `->` Type ] .

GenericBinding = IDENT `=` Type .
GenericBindings = GenericBinding { `,` GenericBinding } .
GenericArgs = `<` [ TypeList ] [ GenericBindings ] `>` .

PathTypeSeg = IDENT [ `::` ] [ GenericArgs | PathTypeFn ] .
PathInType = [ `::` ] PathTypeSeg { `::` PathTypeSeg } .
QualifiedPathInType = `<` Type [ `as` PathInType ] `>` `::` PathTypeSeg { `::` PathTypeSeg } .

PathExprSeg = IDENT [ `::` GenericArgs ] .
PathInExpr = [ `::` ] PathExprSeg { `::` PathExprSeg } .
QualifiedPathInExpr = `<` Type [ `as` PathInType ] `>` `::` PathExprSeg { `::` PathExprSeg } .

## Pats

LiteralPat = BOOLEAN_LITERAL | CHAR_LITERAL | BYTE_LITERAL | STRING_LITERAL | RAW_STRING_LITERAL | BYTE_STRING_LITERAL | RAW_BYTE_STRING_LITERAL | ( [ `-` ] INTEGER_LITERAL | FLOAT_LITERAL ) .

GroupedPat = `(` Pat `)` .

IdentifierPat = [ `ref` ] [ `mut` ] IDENT [ `@` Pat ] .

WildcardPat = `_` .

RestPat = `..` .

TuplePatItems = ( Pat `,` ) | RestPat | ( Pat `,` Pat { `,` Pat } [ `,` ] ) .
TuplePat = `(` [ TuplePatItems ] `)` .

SlicePatItems = Pat { `,` Pat } [ `,` ] .
SlicePat = `[` [ SlicePatItems ] `]` .

StructPatEtc = `..` .
StructPatField = ( TUPLE_INDEX `:` Pat ) | ( IDENT `:` Pat ) | ( [ `ref` ] [ `mut` ] IDENT ) .
StructPatFields = StructPatField { `,` StructPatField } .
StructPatElements = [ StructPatFields ] [ `,` StructPatEtc ] .
StructPat = PathInExpr `{` [ StructPatElements ] `}` .

TupleStructItems = Pat { `,` Pat } [ `,` ] .
TupleStructPat = PathInExpr `(` [ TupleStructItems ] `)` .

PathPat = PathInExpr | QualifiedPathInExpr | StructPat | TupleStructPat .

PrimaryPat = LiteralPat | GroupedPat | IdentifierPat | WildcardPat | RestPat | PathPat .

ReferencePat = { `&` | `&&` [ `mut` ] } PrimaryPat .

RangePatBound = CHAR_LITERAL | BYTE_LITERAL | PathInExpr | QualifiedPathInExpr ( [ `-` ] INTEGER_LITERAL | FLOAT_LITERAL ) .
RangePat = RangePatBound `..=` RangePatBound .

Pat = ReferencePat | RangePat .

## Types

ParenthesizedType = `(` Type `)` .

InferredType = `_` .

NeverType = `!` .

SliceType = `[` Type `]` .

ArrayType = [ Type `;` Expr ] .

TupleType = `(` [ TypeList ] `)` .

ReferenceType = `&` [ `mut` ] Type .

RawPointerType = `*` `mut` | `const` Type .

MaybeNamedParam = [ IDENT | `_` `:` ] Type .
MaybeNamedFunctionParameters = MaybeNamedParam { `,` MaybeNamedParam } [ `,` ] .
MaybeNamedFunctionParametersVariadic = { MaybeNamedParam `,` } MaybeNamedParam `,` `...` .
BareFunctionType = `fn` `(` [ MaybeNamedFunctionParameters | MaybeNamedFunctionParametersVariadic ] `)` [ `->` Type ] .

Type = ParenthesizedType | PathInType | TupleType | NeverType | RawPointerType | ReferenceType | ArrayType | SliceType | InferredType | QualifiedPathInType | BareFunctionType .

## Stats and Exprs

LiteralExpr = CHAR_LITERAL | STRING_LITERAL | RAW_STRING_LITERAL | BYTE_LITERAL | BYTE_STRING_LITERAL | RAW_BYTE_STRING_LITERAL | INTEGER_LITERAL | FLOAT_LITERAL | BOOLEAN_LITERAL .
PrimaryExpr = LiteralExpr | PathInExpr | QualifiedPathInExpr .

ArrayExpr = `[` [ ( Expr { `,` Expr } [ `,` ] ) | ( Expr `;` Expr ) ] `]` .
TupleExpr = `(` [ Expr { `,` Expr } [ `,` ] ] `)` .
GroupedExpr = `(` Expr `)` .
StructExprField = IDENT | ( IDENT | TUPLE_INDEX `:` Expr ) .
StructExprFields = StructExprField { `,` StructExprField } [ `,` `..` Expr [ `,` ] ] .
StructExprStruct = PrimaryExpr `{` [ StructExprFields | `..` Expr ] `}` .
StructExprTuple = PrimaryExpr `(` [ Expr { `,` Expr } [ `,` ] ] `)` .
StructExprUnit = PrimaryExpr .
EnumExprField = IDENT | ( IDENT | TUPLE_INDEX `:` Expr ) .
EnumExprFields = EnumExprField { `,` EnumExprField } [ `,` ] .
EnumExprStruct = PrimaryExpr `{` [ EnumExprFields ] `}` .
EnumExprTuple = PrimaryExpr `(` [ Expr { `,` Expr } [ `,` ] ] `)` .
EnumExprFieldless = PrimaryExpr .
DirectExpr = ArrayExpr | TupleExpr | GroupedExpr | StructExprStruct | StructExprTuple | StructExprUnit | EnumExprStruct | EnumExprTuple | EnumExprFieldless .

TupleIndexingExpr = DirectExpr { `.` TUPLE_INDEX } .

FieldExpr = TupleIndexingExpr { `.` IDENT } .

IndexExpr = FieldExpr { `[` Expr `]` } .

CallParams = Expr { `,` Expr } [ `,` ] .
CallExpr = IndexExpr { `(` [ CallParams ] `)` } .

ErrorPropagationExpr = CallExpr { `?` } .

BorrowExpr = [ `&` | `&&` [ `mut` ] ] ErrorPropagationExpr .

DereferenceExpr = { `*` } BorrowExpr .

NegationExpr = { `-` | `!` } DereferenceExpr .

TypeCastExpr = NegationExpr { `as` Type } .

MulExpr = TypeCastExpr { `*` | `/` | `%` TypeCastExpr } .

AddExpr = MulExpr { `+` | `-` MulExpr } .

ShiftExpr = AddExpr { `<<` | `>>` AddExpr } .

AndExpr = ShiftExpr { `&` ShiftExpr } .

XorExpr = AndExpr { `^` AndExpr } .

OrExpr = XorExpr { `|` XorExpr } .

ComparisonExpr = OrExpr { `==` | `!=` | `>` | `<` | `>=` | `<=` OrExpr } .

LogAndExpr = ComparisonExpr { `&&` ComparisonExpr } .

LogOrExpr = LogAndExpr { `||` LogAndExpr } .

RangeFromToExpr = LogOrExpr [ `..` LogOrExpr ] .
RangeFromExpr = LogOrExpr [ `..` ] .
RangeToExpr = `..` LogOrExpr .
RangeFullExpr = `..` .
RangeInclusiveExpr = LogOrExpr [ `..=` LogOrExpr ] .
RangeToInclusiveExpr = `..=` LogOrExpr .
RangeExpr = RangeFromToExpr | RangeFromExpr | RangeToExpr | RangeFullExpr | RangeInclusiveExpr | RangeToInclusiveExpr .

AssignmentExpr = RangeExpr [ `=` | `+=` | `-=` | `*=` | `/=` | `%=` | `&=` | `|=` | `^=` | `<<=` | `>>=` RangeExpr ] .

ContinueExpr = `continue` .

BreakExpr = `break` [ Expr ] .

ReturnExpr = `return` .

ExprWithoutBlock = AssignmentExpr | ContinueExpr | BreakExpr | ReturnExpr .

BlockExpr = `{` { Stat } [ ExprWithoutBlock ] `}` .

InfiniteLoopExpr = `loop` BlockExpr .

PredicateLoopExpr = `while` Expr BlockExpr .

MatchArmPats = [ `|` ] Pat { `|` Pat } .
PredicatePatLoopExpr = `while` `let` MatchArmsPats `=` Expr BlockExpr .

IteratorLoopExpr = `for` Pat `in` Expr BlockExpr .

IfExpr = `if` Expr BlockExpr [ `else` BlockExpr | IfExpr | IfLetExpr ] .

IfLetExpr = `if` `let` MatchArmPats `=` Expr BlockExpr [ `else` BlockExpr | IfExpr | IfLetExpr ] .

MatchArm = { OuterAttribute } MatchArmPats [ `if` Expr ] `=>` Expr .
MatchExpr = `match` Expr `{` [ { MatchArm `,` } MatchArm [ `,` ] ] `}` .

ExprWithBlock = BlockExpr | InfiniteLoopExpr | PredicateLoopExpr | PredicatePatLoopExpr | IteratorLoopExpr | IfExpr | IfLetExpr | MatchExpr .

Expr = ExprWithoutBlock | ExprWithBlock .

ExprStat = Expr [ `;` ] .

LetStat = `let` Pat [ `:` Type ] [ `=` Expr ] `;` .

Stat = `;` | Item | LetStat | ExprStat .

## Items

Generics = `<` [ IDENT { `,` IDENT } [ `,` ] ] `>` .

Module = `mod` IDENT `;` | ( `{` { Item } `}` ) .

Param = Pat `:` Type .
Params = Param { `,` Param } [ `,` ] .
Function = `fn` IDENT [ Generics ] `(` [ Params ] `)` [ `->` Type ] BlockExpr .

Alias = `type` IDENT [ Generics ] `=` Type `;` .

StructField = IDENT `:` Type .
StructFields = StructField { `,` StructField } [ `,` ] .
StructStruct = `struct` IDENT [ Generics ] ( `{` [ StructFields ] `}` ) | `;` .

TupleFields = Type { `,` Type } [ `,` ] .
TupleStruct = `struct` IDENT [ Generics ] `(` [ TupleFields ] `)` `;` .

EnumItemTuple = `(` [ TupleFields ] `)` .
EnumItemStruct = `{` [ StructFields ] `}` .
EnumItemDiscriminant = `=` Expr .
EnumItem = IDENT [ EnumItemTuple | EnumItemStruct | EnumItemDiscriminant ] .
EnumItems = EnumItem { `,` EnumItem } [ `,` ] .
Enum = `enum` IDENT [ Generics ] `{` [ EnumItems ] `}` .

Union = `union` IDENT [ Generics ] `{` StructFields `}` .

Constant = `const` IDENT | `_` `:` Type `=` Expr `;` .

Static = `static` [ `mut` ] IDENT `:` Type `=` Expr `;` .

Item = Module | Function | Alias | StructStruct | TupleStruct | Enum | Union | Constant | Static .
