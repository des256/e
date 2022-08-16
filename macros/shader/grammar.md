# Reduced Rust Grammar

Ultimately we want to describe shaders in Rust, so this grammar is without macros, use declarations, modules, attributes, lifetimes, traits, closures, where clauses and generics are only references to types outside the shader.

Also, the Rust compiler itself is compiling the shader as well, so this parser will not have to fix errors, only generate a parse tree.

## Lexical Structure

Visibility = `pub` [ `(` `crate` | `self` | `super` | ( `in` SimplePath ) `)` ] .

PathInExpression = [ `::` ] PathExprSegment { `::` PathExprSegment } .

PathExprSegment = PathIdentSegment [ `::` GenericArgs ] .

PathIdentSegment = IDENTIFIER | `super` | `self` | `Self` .

GenericArgs = `<` [ Type { `,` Type } [ `,` ] ] [ IDENTIFIER `=` Type { `,` IDENTIFIER `=` Type } ] `>` .

QualifiedPathInExpression = QualifiedPathType `::` PathExprSegment { `::` PathExprSegment } .

QualifiedPathType = `<` Type [ `as` TypePath ] `>` .

QualifiedPathInType = QualifiedPathType `::` TypePathSegment { `::` TypePathSegment } .

TypePath = [ `::` ] TypePathSegment { `::` TypePathSegment } .

TypePathSegment = PathIdentSegment [ `::` ] [ GenericArgs | TypePathFn ] .

TypePathFn = `(` [ Type { `,` Type } [ `,` ] ] `)` [ `->` Type ] .

## Items

Item = [ Visibility ] Function | TypeAlias | Struct | Enumeration | Union | ConstantItem | StaticItem | Implementation .

Function = [ `async` | `const` ] `fn` IDENTIFIER `(` [ FunctionParameters ] `)` [ `->` Type ] BlockExpression .

FunctionParameters = FunctionParam { `,` FunctionParam } [ `,` ] .

FunctionParam = Pattern `:` Type .

TypeAlias = `type` IDENTIFIER `=` Type `;` .

Struct = StructStruct | TupleStruct .

StructStruct = `struct` IDENTIFIER ( `{` [ StructFields ] `}` ) | `;` .

TupleStruct = `struct` IDENTIFIER `(` [ TupleFields ] `)` `;` .

StructFields = StructField { `,` StructField } [ `,` ] .

StructField = [ Visibility ] IDENTIFIER `:` Type .

TupleFields = TupleField { `,` TupleField } [ `,` ] .

TupleField = [ Visibility ] Type .

Enumeration = `enum` IDENTIFIER `{` [ EnumItems ] `}` .

EnumItems = EnumItem { `,` EnumItem } [ `,` ] .

EnumItem = [ Visibility ] IDENTIFIER [ EnumItemTuple | EnumItemStruct | ( `=` Expression ) ] .

EnumItemTuple = `(` [ TupleFields ] `)` .

EnumItemStruct = `{` [ StructFields ] `}` .

Union = `union` IDENTIFIER `{` StructFields `}` .

ConstantItem = `const` IDENTIFIER | `_` `:` Type `=` Expression `;` .

StaticItem = `static` [ `mut` ] IDENTIFIER `:` Type `=` Expression `;` .

Implementation = `impl` Type `{` { [ Visibility ] ConstantItem | Function | Method } `}` .

Method = [ `async` | `const` ] `fn` IDENTIFIER `(` SelfParam { `,` FunctionParam } [ `,` ] ) [ `->` Type ] BlockExpression .

SelfParam = { OuterAttribute } ShorthandSelf | TypedSelf .

ShorthandSelf = [ `&` ] [ `mut` ] `self` .

TypedSelf = [ `mut` ] `self` `:` Type .

## Statements and Expressions

Statement = `;` | Item | LetStatement | ExpressionStatement .

LetStatement = `let` Pattern [ `:` Type ] [ `=` Expression ] `;` .

ExpressionStatement = ExpressionWithoutBlock | ExpressionWithBlock [ `;` ] .

ExpressionWithoutBlock = LiteralExpression | PathExpression | OperatorExpression | GroupedExpression | ArrayExpression | AwaitExpression | IndexExpression | TupleExpression | TupleIndexingExpression | StructExpression | EnumerationVariantExpression | CallExpression | MethodCallExpression | FieldExpression | ContinueExpression | BreakExpression | RangeExpression | ReturnExpression .

ExpressionWithBlock = BlockExpression | AsyncBlockExpression | UnsafeBlockExpression | LoopExpression | IfExpression | IfLetExpression | MatchExpression .

LiteralExpression = CHAR_LITERAL | STRING_LITERAL | RAW_STRING_LITERAL | BYTE_LITERAL | BYTE_STRING_LITERAL | RAW_BYTE_STRING_LITERAL | INTEGER_LITERAL | FLOAT_LITERAL | BOOLEAN_LITERAL .

PathExpression = PathInExpression | QualifiedPathInExpression .

BlockExpression = `{` [ ( Statement { Statement } [ ExpressionWithoutBlock ] ) | ExpressionWithoutBlock ] `}` .

AsyncBlockExpression = `async` [ `move` ] BlockExpression .

UnsafeBlockExpression = `unsafe` BlockExpression .

OperatorExpression = BorrowExpression | DereferenceExpression | ErrorPropagationExpression | NegationExpression | ArithmeticOrLogicalExpression | ComparisonExpression | LazyBooleanExpression | TypeCastExpression | AssignmentExpression | CompoundAssignmentExpression .

BorrowExpression = `&` | `&&` [ `mut` ] Expression .

DereferenceExpression = `*` Expression .

ErrorPropagationExpression = Expression `?` .

NegationExpression = `-` | `!` Expression .

ArithmeticOrLogicalExpression = Expression `+` | `-` | `*` | `/` | `%` | `&` | `|` | `^` | `<<` | `>>` Expression .

ComparisonExpression = Expression `==` | `!=` | `>` | `<` | `>=` | `<=` Expression .

LazyBooleanExpression = Expression `||` | `&&` Expression .

TypeCastExpression = Expression `as` Type .

AssignmentExpression = Expression `=` Expression .

CompoundAssignmentExpression = Expression `+=` | `-=` | `*=` | `/=` | `%=` | `&=` | `|=` | `^=` | `<<=` | `>>=` Expression .

GroupedExpression = `(` Expression `)` .

ArrayExpression = `[` [ ArrayElements ] `]` .

ArrayElements = ( Expression { `,` Expression } [ `,` ] ) | ( Expression `;` Expression ) .

IndexExpression = Expression `[` Expression `]` .

TupleExpression = `(` [ Expression { `,` Expression } [ `,` ] ] `)` .

TupleIndexingExpression = Expression `.` TUPLE_INDEX .

StructExpression = StructExprStruct | StructExprTuple | StructExprUnit .

StructExprStruct = PathInExpression `{` [ StructExprFields | StructBase ] `}` .

StructExprFields = StructExprField { `,` StructExprField } [ `,` StructBase [ `,` ] ] .

StructExprField = IDENTIFIER | ( IDENTIFIER | TUPLE_INDEX `:` Expression ) .

StructBase = `..` Expression .

StructExprTuple = PathInExpression `(` [ Expression { `,` Expression } [ `,` ] ] `)` .

StructExprUnit = PathInExpression .

EnumVariantExpression = EnumExprStruct | EnumExprTuple | EnumExprFieldless .

EnumExprStruct = PathInExpression `{` [ EnumExprFields ] `}` .

EnumExprFields = EnumExprField { `,` EnumExprField } [ `,` ] .

EnumExprField = IDENTIFIER | ( IDENTIFIER | TUPLE_INDEX `:` Expression ) .

EnumExprTuple = PathInExpression `(` [ Expression { `,` Expression } [ `,` ] ] `)` .

EnumExprFieldless = PatthInExpression .

CallExpression = Expression `(` [ CallParams ] `)` .

CallParams = Expression { `,` Expression } [ `,` ] .

MethodCallExpression = Expression `.` PathExprSegment `(` [ CallParams ] `)` .

FieldExpression = Expression `.` IDENTIFIER .

LoopExpression = [ LoopLabel ] InfiniteLoopExpression | PredicateLoopExpression | PredicatePatternLoopExpression | IteratorLoopExpression .

InfiniteLoopExpression = `loop` BlockExpression .

PredicateLoopExpression = `while` StructlessExpression BlockExpression .

PredicatePatternLoopExpression = `while` `let` MatchArmsPatterns `=` StructlessLazylessExpression BlockExpression .

IteratorLoopExpression = `for` Pattern `in` StructlessExpression BlockExpression .

LoopLabel = LABEL `:` .

BreakExpression = `break` [ LABEL ] [ Expression ] .

ContinueExpression = `continue` [ LABEL ] .

RangeExpression = RangeExpr | RangeFromExpr | RangeToExpr | RangeFullExpr | RangeInclusiveExpr | RangeToInclusiveExpr .

RangeExpr = Expression `..` Expression .

RangeFromExpr = Expression `..` .

RangeToExpr = `..` Expression .

RangeFullExpression = `..` .

RangeInclusiveExpression = Expression `..=` Expression .

RangeToInclusiveExpression = `..=` Expression .

IfExpression = `if` StructlessExpression BlockExpression [ `else` BlockExpression | IfExpression | IfLetExpression ] .

IfLetExpression = `if` `let` MatchArmPatterns `=` StructlessLazylessExpression BlockExpression [ `else` BlockExpression | IfExpression | IfLetExpression ] .

MatchExpression = `match` StructlessExpression `{` [ MatchArms ] `}` .

MatchArms = { MatchArm `=>` ( ExpressionWithoutBlock `,` ) | ( ExpressionWithBlock `,` ) } MatchArm `=>` Expression [ `,` ] .

MatchArm = { OuterAttribute } MatchArmPatterns [ MatchArmGuard ] .

MatchArmPatterns = [ `|` ] Pattern { `|` Pattern } .

MatchArmGuard = `if` Expression .

ReturnExpression = `return` [ Expression ] .

AwaitExpression = Expression `.` `await` .

## Patterns

Pattern = PatternWithoutRange | RangePattern .

PatternWithoutRange = LiteralPattern | IdentifierPattern | WildcardPattern | RestPattern | ObsoleteRangePattern | ReferencePattern | StructPattern | TupleStructPattern | TuplePattern | GroupedPattern | SlicePattern | PathPattern | MacroInvocation .

LiteralPattern = BOOLEAN_LITERAL | CHAR_LITERAL | BYTE_LITERAL | STRING_LITERAL | RAW_STRING_LITERAL | BYTE_STRING_LITERAL | RAW_BYTE_STRING_LITERAL | ( [ `-` ] INTEGER_LITERAL | FLOAT_LITERAL ) .

IdentifierPattern = [ `ref` ] [ `mut` ] IDENTIFIER [ `@` Pattern ] .

WildcardPattern = `_` .

RestPattern = `..` .

RangePattern = RangePatternBound `..=` RangePatternBound .

ObsoleteRangePattern = RangePatternBound `...` RangePatternBound .

RangePatternBound = CHAR_LITERAL | BYTE_LITERAL | PathInExpression | QualifiedPathInExpression ( [ `-` ] INTEGER_LITERAL | FLOAT_LITERAL ) .

ReferencePattern = `&` | `&&` [ `mut` ] PatternWithoutRange .

StructPattern = PathInExpression `{` [ StructPatternElements ] `}` .

StructPatternElements = [ StructPatternFields ] [ `,` `..` ] .

StructPatternFields = StructPatternField { `,` StructPatternField } .

StructPatternField = ( TUPLE_INDEX `:` Pattern ) | ( IDENTIFIER `:` Pattern ) | ( [ `ref` ] [ `mut` ] IDENTIFIER ) .

TupleStructPattern = PathInExpression `(` [ TupleStructItems ] `)` .

TupleStructItems = Pattern { `,` Pattern } [ `,` ] .

TuplePattern = `(` [ TuplePatternItems ] `)` .

TuplePatternItems = ( Pattern `,` ) | RestPattern | ( Pattern `,` Pattern { `,` Pattern } [ `,` ] ) .

GroupedPattern = `(` Pattern `)` .

SlicePattern = `[` [ SlicePatternItems ] `]` .

SlicePatternItems = Pattern { `,` Pattern } [ `,` ] .

PathPattern = PathInExpression | QualifiedPathInExpression .

## Type System

Type = ParenthesizedType | TypePath | TupleType | NeverType | RawPointerType | ReferenceType | ArrayType | SliceType | InferredType | QualifiedPathInType | BareFunctionType .

ParenthesizedType = `(` Type `)` .

NeverType = `!` .

TupleType = `(` [ Type { `,` Type } [ `,` ] `)` .

ArrayType = [ Type `;` Expression ] .

SliceType = `[` Type `]` .

ReferenceType = `&` [ `mut` ] Type .

RawPointerType = `*` `mut` | `const` Type .

BareFunctionType = [ `async` | `const` ] `fn` `(` [ FunctionParametersMaybeNamedVariadic ] `)` [ `->` Type ] .

FunctionParametersMaybeNamedVariadic = MaybeNamedFunctionParameters | MaybeNamedFunctionParametersVariadic .

MaybeNamedFunctionParameters = MaybeNamedParam { `,` MaybeNamedParam } [ `,` ] .

MaybeNamedParam = [ IDENTIFIER | `_` `:` ] Type .

MaybeNamedFunctionParametersVariadic = { MaybeNamedParam `,` } MaybeNamedParam `,` `...` .

InferredType = `_` .
