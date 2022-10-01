use {
    sr::*,
    std::{
        rc::Rc,
        collections::HashMap,
    }
};

struct Library {
    pub tuples: HashMap<String,Rc<ast::Tuple>>,
    pub structs: HashMap<String,Rc<ast::Struct>>,
    pub enums: HashMap<String,Rc<ast::Enum>>,
    pub aliases: HashMap<String,Rc<ast::Alias>>,
    pub consts: HashMap<String,Rc<ast::Const>>,
    pub functions: HashMap<String,Rc<ast::Function>>,
    pub locals: HashMap<String,Rc<ast::Symbol>>,
}

trait ResolveSymbols {
    fn resolve_symbols(&mut self,library: &mut Library);
}

impl ResolveSymbols for ast::Type {
    fn resolve_symbols(&mut self,library: &mut Library) {
        match self {
            ast::Type::Inferred |
            ast::Type::Void |
            ast::Type::Integer |
            ast::Type::Float |
            ast::Type::Bool |
            ast::Type::U8 |
            ast::Type::I8 |
            ast::Type::U16 |
            ast::Type::I16 |
            ast::Type::U32 |
            ast::Type::I32 |
            ast::Type::U64 |
            ast::Type::I64 |
            ast::Type::USize |
            ast::Type::ISize |
            ast::Type::F16 |
            ast::Type::F32 |
            ast::Type::F64 => { },
            ast::Type::AnonTuple(types) => {
                for type_ in types.iter() {
                    type_.resolve_symbols(library);
                }
            },
            ast::Type::Array(type_,expr) => {
                type_.resolve_symbols(library);
                expr.resolve_symbols(library);
            },
            ast::Type::UnknownIdent(ident) => {
                if library.tuples.contains_key(ident) {
                    *self = ast::Type::Tuple(Rc::clone(&library.tuples[ident]));
                }
                else if library.structs.contains_key(ident) {
                    *self = ast::Type::Struct(Rc::clone(&library.structs[ident]));
                }
                else if library.enums.contains_key(ident) {
                    *self = ast::Type::Enum(Rc::clone(&library.enums[ident]));
                }
                else if library.aliases.contains_key(ident) {
                    *self = ast::Type::Alias(Rc::clone(&library.aliases[ident]));
                }
                else {
                    panic!("unknown type {}",ident);
                }
            },
            ast::Type::Tuple(_) |
            ast::Type::Struct(_) |
            ast::Type::Enum(_) |
            ast::Type::Alias(_) => { },
        }
    }
}

impl ResolveSymbols for ast::Pat {
    fn resolve_symbols(&mut self,library: &mut Library) {
        match self {
            ast::Pat::Wildcard |
            ast::Pat::Rest |
            ast::Pat::Boolean(_) |
            ast::Pat::Integer(_) |
            ast::Pat::Float(_) => { },
            ast::Pat::AnonTuple(pats) => {
                for pat in pats.iter() {
                    pat.resolve_symbols(library);
                }
            },
            ast::Pat::Array(pats) => {
                for pat in pats.iter() {
                    pat.resolve_symbols(library);
                }
            },
            ast::Pat::Range(pat,pat2) => {
                pat.resolve_symbols(library);
                pat2.resolve_symbols(library);
            },
            ast::Pat::UnknownIdent(_) => { },
            ast::Pat::UnknownTuple(ident,pats) => {
                if library.tuples.contains_key(ident) {
                    *self = ast::Pat::Tuple(Rc::clone(&library.tuples[ident]),pats.clone());
                }
                else {
                    panic!("unknown tuple {}",ident);
                }
            },
            ast::Pat::UnknownStruct(ident,ident_pats) => {
                if library.structs.contains_key(ident) {
                    // TODO: convert ident_pats to Vec<IndexPat>
                    *self = ast::Pat::Structs(Rc::clone(&library.structs[ident]),TODO);
                }
                else {
                    panic!("unknown struct {}",ident);
                }
            }
            ast::Pat::UnknownVariant(ident,variant) => {
                if library.enums.contains_key(ident) {
                    // TODO: convert variant to PatVariant
                    *self = ast::Pat::Variant(Rc::clone(&library.enums[ident]),TODO);
                }
            },
            ast::Pat::Tuple(_,pats) => {
                for pat in pats.iter() {
                    pat.resolve_symbols(library);
                }
            },
            ast::Pat::Struct(_,index_pats) => {
                for index_pat in index_pats.iter() {
                    if let ast::IndexPat::IndexPat(_,pat) = index_pat {
                        pat.resolve_symbols(libary);
                    }
                }
            },
            ast::Pat::Variant(_,variant) => {
                match variant {
                    ast::PatVariant::Tuple(_,pats) => {
                        for pat in pats.iter() {
                            pat.resolve_symbols(library);
                        }
                    },
                    ast::PatVariant::Struct(_,index_pats) => {
                        for index_pat in index_pats.iter() {
                            if let ast::IndexPat::IndexPat(_,pat) = index_pat {
                                pat.resolve_symbols(library);
                            }
                        }
                    },
                    _ => { },
                }
            },
        }
    }
}

impl ResolveSymbols for ast::Expr {
    fn resolve_symbols(&mut self,library: &mut Library) {
        match self {
            Expr::Boolean(_) |
            Expr::Integer(_) |
            Expr::Float(_) => { },
            Expr::Array(exprs) => {
                for expr in exprs.iter() {
                    expr.resolve_symbols(library);
                }
            },
            Expr::Cloned(expr,expr2) => {
                expr.resolve_symbols(library);
                expr2.resolve_symbols(library);
            },
            Expr::Index(expr,expr2) => {
                expr.resolve_symbols(library);
                expr2.resolve_symbols(library);
            },
            Expr::Cast(expr,type_) => {
                expr.resolve_symbols(library);
                type_.resolve_symbols(library);
            },
            Expr::AnonTuple(exprs) => {
                for expr in exprs.iter() {
                    expr.resolve_symbols(library);
                }
            },
            Expr::Unary(op,expr) => expr.resolve_symbols(library),
            Expr::Binary(expr,op,expr2) => {
                expr.resolve_symbols(library);
                expr2.resolve_symbols(library);
            },
            Expr::Continue => { },
            Expr::Break(expr) => if let Some(expr) = expr {
                expr.resolve_symbols(library);
            },
            Expr::Return(expr) => if let Some(expr) = expr {
                expr.resolve_symbols(library);
            },
            Expr::Block(block) => block.resolve_symbols(library),
            Expr::If(expr,block,else_expr) => {
                expr.resolve_symbols(library);
                block.resolve_symbols(library);
                if let Some(else_expr) = else_expr {
                    else_expr.resolve_symbols(library);
                }
            },
            Expr::Loop(block) => block.resolve_symbols(library),
            Expr::While(expr,block) => {
                expr.resolve_symbols(library);
                block.resolve_symbols(library);
            },
            Expr::IfLet(pats,expr,block,else_expr) => {
                for pat in pats.iter() {
                    pat.resolve_symbols(library);
                }
                expr.resolve_symbols(library);
                block.resolve_symbols(library);
                if let Some(else_expr) = else_expr {
                    else_expr.resolve_symbols(library);
                }
            },
            Expr::For(pats,range,block) => {
                for pat in pats.iter() {
                    pat.resolve_symbols(library);
                }
                range.resolve_symbols(library);
                block.resolve_symbols(library);
            },
            Expr::WhileLet(pats,expr,block) => {
                for pat in pats.iter() {
                    pat.resolve_symbols(library);
                }
                expr.resolve_symbols(library);
                block.resolve_symbols(library);
            },
            Expr::Match(expr,arms) => {
                expr.resolve_symbols(library);
                for (pats,if_expr,expr) in arms.iter() {
                    for pat in pats.iter() {
                        pat.resolve_symbols(library);
                    }
                    if let Some(if_expr) = else_expr {
                        if_expr.resolve_symbols(library);
                    }
                    expr.resolve_symbols(library);                    
                }
            },
            Expr::UnknownIdent(ident) => {
                // TODO: Param, Local, Const
            },
            Expr::UnknownTupleOrCall(ident,exprs) => {
                // TODO: Tuple, Call
            },
            Expr::UnknownStruct(ident,fields) => {
                // TODO: Struct
            },
            Expr::UnknownVariant(ident,variant) => {
                // TODO: Variant
            },
            Expr::UnknownMethod(expr,ident,exprs) => {
                // TODO: Method
            },
            Expr::UnknownField(expr,ident) => {
                // TODO: Field
            },
            Expr::UnknownTupleIndex(expr,index) => {
                // TODO: TupleIndex
            }
        }
    }
}

impl ResolveSymbols for ast::Stat {
    fn resolve_symbols(&mut self,library: &mut Library) {
    }
}

impl ResolveSymbols for ast::Block {
    fn resolve_symbols(&mut self,library: &mut Library) {
    }
}

impl ResolveSymbols for ast::Tuple {
    fn resolve_symbols(&mut self,library: &mut Library) {
    }
}

impl ResolveSymbols for ast::Struct {
    fn resolve_symbols(&mut self,library: &mut Library) {
    }
}

impl ResolveSymbols for ast::Enum {
    fn resolve_symbols(&mut self,library: &mut Library) {
    }
}

impl ResolveSymbols for ast::Const {
    fn resolve_symbols(&mut self,library: &mut Library) {
    }
}

impl ResolveSymbols for ast::Function {
    fn resolve_symbols(&mut self,library: &mut Library) {
    }
}

impl ResolveSymbols for ast::Method {
    fn resolve_symbols(&mut self,library: &mut Library) {
    }
}

impl ResolveSymbols for ast::Alias {
    fn resolve_symbols(&mut self,library: &mut Library) {
    }
}

impl ResolveSymbols for ast::Module {
    fn resolve_symbols(&mut self,library: &mut Library) {
    }
}
