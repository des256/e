# Revised Reduced Rust Grammar

## Shader

shader = module_def | function_def .

## Commonly Used

ident_types = [ IDENT ':' type { ',' IDENT ':' type } [ ',' ] ] .

paren_ident_types = '(' ident_types ')' .

brace_ident_types = '{' ident_types '}' .

types = [ type { ',' type } [ ',' ] ] .

paren_types = '(' types ')' .

exprs = [ expr { ',' expr } [ ',' ] ] .

paren_exprs = '(' exprs ')' .

bracket_exprs = '[' exprs ']' .

brace_field_exprs = '{' [ FIELD ':' expr { ',' FIELD ':' expr } [ ',' ] ] '}' .

brace_field_pats = '{' [ ( FIELD [ ':' pat ] | '_' { ',' FIELD [ ':' pat ] | '_' } [ ',' [ '..' ] ] ) | '..' ] '}' .

paren_pats = '(' [ ( pat | '_' { ',' pat | '_' } [ ',' [ '..' ] ] ) | '..' ] ')' .

bracket_pats = '[' { pat [ ',' ] } ']' .

## Items

function_def = 'fn' IDENT paren_ident_types [ '->' type ] block .

struct_def = 'struct' IDENT brace_ident_types .

tuple_def = 'struct' IDENT paren_types .

variant = IDENT [ brace_ident_types | paren_types | ( '=' NUMBER ) ] .

enum_def = 'enum' IDENT '{' [ variant { ',' variant } [ ',' ] ] '}' .

const_def = 'const' IDENT ':' type '=' expr .

module_def = 'mod' IDENT '{' { function_def | struct_def | tuple_def | enum_def | const_def } '}' .

## Literals and Identifiers

BOOLEAN = 'true' | 'false' .    

NUMBER = numeric literal

FIELD = field name

GLOBAL = global variable name

LOCAL = local variable name

PARAM = function parameter name

CONST = global constant name

FUNCTION = function name

BASETYPE = base shader type (see elsewhere)

STRUCT = name of a struct

TUPLE = name of a tuple

ENUM = name of an enum

## Types

type = 
    '_' |
    BASETYPE |
    STRUCT |
    TUPLE |
    ENUM |
    ( '[' type ']' ) |
    paren_types .

## Patterns

pat =
    '_' |
    BOOLEAN |
    NUMBER |
    CONST |
    ( IDENT [ brace_field_pats | paren_pats ] ) |
    bracket_pats |
    paren_pats |
    ( ENUM '::' VARIANT [ brace_field_pats | paren_pats ] ) .

ranged_pat = pat [ '..=' pat ] .

pats = [ '|' ] ranged_pat { '|' ranged_pat } .

## Expressions

array = bracket_exprs .

cloned_array = '[' expr ';' expr ']' .

struct = STRUCT brace_field_exprs .

tuple = TUPLE paren_exprs .

anon_tuple = paren_exprs .

struct_enum = brace_field_exprs .

tuple_enum = paren_exprs .

enum = ENUM '::' VARIANT [ struct_enum | tuple_enum ] .

call = FUNCTION paren_exprs .

primary =
    BOOLEAN |
    NUMBER |
    GLOBAL |
    LOCAL |
    PARAM |
    CONST |
    array |
    cloned_array |
    struct |
    tuple |
    anon_tuple |
    enum |
    call .

postfix_field = '.' FIELD .

pustfix_tuple = '.' NUMBER .

postfix_index = '[' expr ']' .

postfix_cast = 'as' type .

postfix = primary {
    postfix_field |
    postfix_tuple |
    postfix_index |
    postfix_cast
} .

prefix = { '-' | '!' } postfix .

mul = prefix { '*' | '/' | '%' prefix } .

add = mul { '+' | '-' mul } .

shift = add { '<<' | '>>' add } .

and = shift { '&' shift } .

or = and { '|' and } .

xor = or { '^' or } .

comp = xor { '==' | '!=' | '>' | '<' | '>=' | '<=' xor } .

logand = comp { '&&' comp } .

logor = logand { '||' logand } .

assign = logor { '=' | '+=' | '-=' | '*=' | '/=' | '%=' | '&=' | '|=' | '^=' | '<<=' | '>>=' logor } .

continue = 'continue' .

break = 'break' [ expr ] .

return = 'return' [ expr ] .

block = '{' { stat } [ expr ] '}' .

if = 'if' expr block [ 'else' else ] .

if_let = 'if' 'let' pats '=' expr block [ 'else' else ] .

else = block | if | if_let .

loop = 'loop' block .

range_from_to = expr '..' expr .

range_from_to_incl = expr '..=' expr .

range_from = expr '..' .

range_to = '..' expr .

range_to_incl = '..=' expr .

range = '..' .

for = 'for' pats 'in' range block .

while = 'while' expr block .

while_let = 'while' 'let' pats '=' expr block .

match = 'match' expr '{' { pats [ 'if' expr ] '=>' expr [ ',' ] } '}' .

expr =
    assign |
    continue |
    break |
    return |
    block |
    if |
    if_let |
    loop |
    for |
    while |
    while_let |
    match .

## Statements

let = 'let' pat [ ':' type ] '=' expr ';' .

expr_stat = expr ';' .

stat = let | expr_stat .
