use {
    crate::*,
    std::fmt::{
        Display,
        Formatter,
        Result,
    },
};

impl Display for Item {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Item::Mod(ident,items) => write!(f,"mod {} {{ ... }}",ident),
            Item::Func(ident,args,result,block) => write!(f,"fn {}(...) ... {{ ... }}",ident),
            Item::Alias(ident,ty) => write!(f,"type {} = ...",ident),
            Item::Struct(ident,fields) => write!(f,"struct {} {{ ... }}",ident),
            Item::Tuple(ident,tys) => write!(f,"struct {} ( ... )",ident),
            Item::Enum(ident,vars) => write!(f,"enum {} {{ ... }}",ident),
            Item::Union(ident,fields) => write!(f,"union {} {{ ... }}",ident),
            Item::Const(ident,ty,expr) => write!(f,"const {}: ... = ...",if let Some(ident) = ident { ident } else { "_" }),
            Item::Static(is_mut,ident,ty,expr) => write!(f,"static ... {}: ... = ...",ident),
            Item::Impl(ty,items) => write!(f,"impl ... {{ ... }}"),
        }
    }
}
