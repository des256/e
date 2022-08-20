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
            Item::Module(ident,items) => {
                write!(f,"mod {} {{ ",ident);
                for item in items {
                    write!(f,"{} ",item);
                }
                write!(f,"}}")
            },
            Item::Function(ident,generics,params,ty,block) => {
                write!(f,"fn {}",ident);
                if generics.len() > 0 {
                    write!(f,"<");
                    for ident in generics {
                        write!(f,"{},",ident);
                    }
                    write!(f,">");
                }
                write!(f,"(");
                for param in params {
                    write!(f,"{}: {},",param.pat,param.ty)
                }
                write!(f,") ");
                if let Some(ty) = ty {
                    write!(f,"-> {}",ty);
                }
                write!(f,"{{ {} }}",block)
            },
            Item::Alias(ident,generics,ty) => {
                write!(f,"type {}",ident);
                if generics.len() > 0 {
                    write!(f,"<");
                    for ident in generics {
                        write!(f,"{},",ident);
                    }
                    write!(f,">");
                }
                write!(f," = {}",ty)
            },
            Item::Struct(ident,generics,fields) => {
                write!(f,"struct {}",ident);
                if generics.len() > 0 {
                    write!(f,"<");
                    for ident in generics {
                        write!(f,"{},",ident);
                    }
                    write!(f,">");
                }
                if fields.len() > 0 {
                    write!(f," {{ ");
                    for field in fields {
                        write!(f,"{}: {},",field.ident,field.ty)
                    }
                    write!(f,"}}")
                }
                else {
                    write!(f,";")
                }
            },
            Item::Tuple(ident,generics,types) => {
                write!(f,"struct {}",ident);
                if generics.len() > 0 {
                    write!(f,"<");
                    for ident in generics {
                        write!(f,"{},",ident);
                    }
                    write!(f,">");
                }
                write!(f,"(");
                for ty in types {
                    write!(f,"{},",ty);
                }
                write!(f,")")
            },
            Item::Enum(ident,generics,variants) => {
                write!(f,"enum {}",ident);
                if generics.len() > 0 {
                    write!(f,"<");
                    for ident in generics {
                        write!(f,"{},",ident);
                    }
                    write!(f,">");
                }
                write!(f," {{ ");
                for variant in variants {
                    write!(f,"{}, ",variant);
                }
                write!(f,"}}")
            },
            Item::Union(ident,generics,fields) => {
                write!(f,"union {}",ident);
                if generics.len() > 0 {
                    write!(f,"<");
                    for ident in generics {
                        write!(f,"{},",ident);
                    }
                    write!(f,">");
                }
                if fields.len() > 0 {
                    write!(f," {{ ");
                    for field in fields {
                        write!(f,"{}: {}, ",field.ident,field.ty);
                    }
                    write!(f,"}}")
                }
                else {
                    write!(f,";")
                }
            },
            Item::Const(ident,ty,expr) => {
                write!(f,"const {}: {} = {};",if let Some(ident) = ident { ident } else { "_" },ty,expr)
            },
            Item::Static(is_mut,ident,ty,expr) => {
                write!(f,"static ");
                if *is_mut {
                    write!(f,"mut ");
                }
                write!(f,"{}: {} = {};",ident,ty,expr)
            },
        }
    }
}

impl Display for Variant {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Variant::Naked(ident) => write!(f,"{}",ident),
            Variant::Discr(ident,expr) => write!(f,"{} = {}",ident,expr),
            Variant::Struct(ident,fields) => {
                write!(f,"{} {{",ident);
                for field in fields {
                    write!(f,"{}: {}, ",field.ident,field.ty);
                }
                write!(f,"}}")
            },
            Variant::Tuple(ident,types) => {
                write!(f,"{}(",ident);
                for ty in types {
                    write!(f,"{},",ty);
                }
                write!(f,")")
            },
        }
    }
}

impl Display for Expr {
    fn fmt(&self,f: &mut Formatter) -> Result { write!(f,"TODO") }
}

impl Display for Type {
    fn fmt(&self,f: &mut Formatter) -> Result { write!(f,"TODO") }
}

impl Display for Pat {
    fn fmt(&self,f: &mut Formatter) -> Result { write!(f,"TODO") }
}
