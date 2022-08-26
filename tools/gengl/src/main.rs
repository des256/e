use {
    std::{
        fs::File,
        io::BufReader,
    },
    xml::{
        reader::{
            EventReader,
            XmlEvent,
        },
        attribute::OwnedAttribute,
    },
};

struct Type {
    name: String,
    ctype: String,
    requires_khrplatform: bool,
}

struct Enum {
    name: String,
    value: String,
}

struct Param {
    name: String,
    ctype: String,
}

struct Function {
    name: String,
    alias: Option<String>,
    ctype: String,
    params: Vec<Param>,
}

struct Main {
    parser: EventReader<BufReader<File>>,
    types: Vec<Type>,
    bitmasks: Vec<Enum>,
    enums: Vec<Enum>,
    functions: Vec<Function>,
}

impl Main {
    fn new(filename: &str) -> Option<Main> {
        let file = File::open(filename).ok()?;
        let file = BufReader::new(file);
        let parser = EventReader::new(file);    
        Some(Main {
            parser,
            types: Vec::new(),
            bitmasks: Vec::new(),
            enums: Vec::new(),
            functions: Vec::new(),
        })
    }

    fn expect_characters(&mut self) -> String {
        if let Ok(xml) = self.parser.next() {
            if let XmlEvent::Characters(string) = xml {
                string
            }
            else {
                panic!("characters expected");
            }
        }
        else {
            panic!("unexpected end of file");
        }
    }

    fn expect_end_element(&mut self) {
        if let Ok(xml) = self.parser.next() {
            if let XmlEvent::EndElement { .. } = xml {
                return;
            }
            else {
                panic!("end element expected");
            }
        }
        else {
            panic!("unexpected end of file");
        }
    }

    fn expect_start_document(&mut self) {
        if let Ok(xml) = self.parser.next() {
            if let XmlEvent::StartDocument { .. } = xml {
                return;
            }
            else {
                panic!("document start expected");
            }
        }
        else {
            panic!("unexpected end of file");
        }
    }

    fn parse_comment(&mut self) {
        let mut depth = 0isize;
        while let Ok(xml) = self.parser.next() {
            match xml {
                XmlEvent::StartElement { name,.. } => depth += 1,
                XmlEvent::EndElement { .. } => {
                    depth -= 1;
                    if depth < 0 {
                        return;
                    }
                },
                _ => { },
            }
        }
        panic!("unexpected end of file");
    }

    fn parse_type(&mut self,attrs: Vec<OwnedAttribute>) -> Type {
        let mut ty = Type {
            name: String::new(),
            ctype: String::new(),
            requires_khrplatform: false,
        };
        for attr in attrs {
            if (attr.name.local_name.as_str() == "requires") && (attr.value.as_str() == "khrplatform") {
                ty.requires_khrplatform = true;
            }
        }

        while let Ok(xml) = self.parser.next() {
            match xml {
                XmlEvent::StartElement { name,.. } => {
                    match name.local_name.as_str() {
                        "name" => {
                            ty.name = self.expect_characters();
                            self.expect_end_element();
                        },
                        "apientry" => {
                            ty.ctype += "APIENTRY";
                            self.expect_end_element();
                        },
                        _ => {
                            panic!("name tag expected in type definition");
                        },
                    }
                },
                XmlEvent::EndElement { .. } => {
                    if ty.ctype.starts_with("typedef ") {
                        ty.ctype = ty.ctype[8..].to_string();
                    }
                    return ty;
                },
                XmlEvent::Characters(string) => {
                    if string.as_str() != ";" {
                        ty.ctype += &string;
                    }
                },
                _ => { },
            }
        }
        panic!("unexpected end of file");
    }

    fn parse_types(&mut self) {
        while let Ok(xml) = self.parser.next() {
            match xml {
                XmlEvent::StartElement { name,attributes,.. } => {
                    match name.local_name.as_str() {
                        "type" => {
                            let ty = self.parse_type(attributes);
                            self.types.push(ty);
                        },
                        _ => { panic!("type expected instead of {}",name.local_name); },
                    }
                },
                XmlEvent::EndElement { .. } => return,
                _ => { },
            }
        }
        panic!("unexpected end of file");
    }

    fn parse_enum(&mut self,attrs: Vec<OwnedAttribute>) -> Enum {
        let mut en = Enum {
            name: String::new(),
            value: String::new(),
        };
        for attr in attrs {
            match attr.name.local_name.as_str() {
                "name" => en.name = attr.value,
                "value" => en.value = attr.value,
                _ => { },
            }
        }
        self.expect_end_element();
        en
    }

    fn parse_enums(&mut self,attrs: Vec<OwnedAttribute>) {
        let mut parsing_bitmasks = false;
        for attr in attrs {
            if (attr.name.local_name.as_str() == "type") && (attr.value.as_str() == "bitmask") {
                parsing_bitmasks = true;
            }
        }
        while let Ok(xml) = self.parser.next() {
            match xml {
                XmlEvent::StartElement { name,attributes,.. } => {
                    match name.local_name.as_str() {
                        "enum" => {
                            let en = self.parse_enum(attributes);
                            if parsing_bitmasks {
                                self.bitmasks.push(en);
                            }
                            else {
                                self.enums.push(en);
                            }
                        },
                        "unused" => self.expect_end_element(),
                        _ => { panic!("enum expected instead of {}",name.local_name); },
                    }
                },
                XmlEvent::EndElement { .. } => {
                    return;
                },
                _ => { },
            }
        }
        panic!("unexpected end of file");
    }

    fn parse_feature(&mut self,attrs: Vec<OwnedAttribute>) {
        /*
        print!("feature(");
        let mut first_attr = true;
        for attr in attrs {
            if !first_attr {
                print!(",");
            }
            print!("{}=\"{}\"",attr.name,attr.value);
            first_attr = false;
        }
        println!("):");
        */
        let mut depth = 0isize;
        while let Ok(xml) = self.parser.next() {
            match xml {
                XmlEvent::StartElement { name,.. } => {
                    //println!("    {}: {}",depth,name);
                    depth += 1;
                },
                XmlEvent::EndElement { .. } => {
                    depth -= 1;
                    if depth < 0 {
                        return;
                    }
                },
                _ => { },
            }
        }
        panic!("unexpected end of file");
    }

    fn parse_extensions(&mut self,attrs: Vec<OwnedAttribute>) {
        /*
        print!("extensions(");
        let mut first_attr = true;
        for attr in attrs {
            if !first_attr {
                print!(",");
            }
            print!("{}=\"{}\"",attr.name,attr.value);
            first_attr = false;
        }
        println!("):");
        */
        let mut depth = 0isize;
        while let Ok(xml) = self.parser.next() {
            match xml {
                XmlEvent::StartElement { name,.. } => {
                    //println!("    {}: {}",depth,name);
                    depth += 1;
                },
                XmlEvent::EndElement { .. } => {
                    depth -= 1;
                    if depth < 0 {
                        return;
                    }
                },
                _ => { },
            }
        }
        panic!("unexpected end of file");
    }

    fn parse_command_proto(&mut self,attrs: Vec<OwnedAttribute>) -> Function {
        let mut function = Function {
            name: String::new(),
            alias: None,
            ctype: String::new(),
            params: Vec::new(),
        };
        let mut depth = 0isize;
        while let Ok(xml) = self.parser.next() {
            match xml {
                XmlEvent::StartElement { name,.. } => {
                    match name.local_name.as_str() {
                        "name" => {
                            function.name = self.expect_characters();
                            self.expect_end_element();
                        },
                        _ => depth += 1,
                    }
                },
                XmlEvent::EndElement { .. } => {
                    depth -= 1;
                    if depth < 0 {
                        return function;
                    }
                },
                XmlEvent::Characters(string) => {
                    if string.as_str() != ";" {
                        function.ctype += &string;
                    }
                },
                _ => { },
            }
        }
        panic!("unexpected end of file");
    }

    fn parse_command_param(&mut self,attrs: Vec<OwnedAttribute>) -> Param {
        let mut param = Param {
            name: String::new(),
            ctype: String::new(),
        };
        while let Ok(xml) = self.parser.next() {
            match xml {
                XmlEvent::StartElement { name,.. } => {
                    match name.local_name.as_str() {
                        "name" => {
                            param.name = self.expect_characters();
                            self.expect_end_element();
                        },
                        "ptype" => {
                            param.ctype += &self.expect_characters();
                            self.expect_end_element();
                        },
                        _ => {
                            panic!("name or ptype expected in command param");
                        },
                    }
                },
                XmlEvent::EndElement { .. } => {
                    match param.name.as_str() {
                        "ref" => param.name = "r#ref".to_string(),
                        "mut" => param.name = "r#mut".to_string(),
                        "type" => param.name ="r#type".to_string(),
                        "in" => param.name = "r#in".to_string(),
                        "box" => param.name = "r#box".to_string(),
                        _ => { },
                    }
                    return param;
                },
                XmlEvent::Characters(string) => {
                    param.ctype += &string;
                },
                _ => { },
            }
        }
        panic!("unexpected end of file");
    }

    fn parse_command(&mut self,attrs: Vec<OwnedAttribute>) -> Function {
        let mut function = Function {
            name: String::new(),
            alias: None,
            ctype: String::new(),
            params: Vec::new(),
        };
        let mut depth = 0isize;
        while let Ok(xml) = self.parser.next() {
            match xml {
                XmlEvent::StartElement { name,attributes,.. } => {
                    match name.local_name.as_str() {
                        "proto" => function = self.parse_command_proto(attributes),
                        "param" => {
                            let mut param = self.parse_command_param(attributes);
                            match param.ctype.as_str() {
                                "void *" => param.ctype = "*mut ()".to_string(),
                                "void **" => param.ctype = "*mut *mut ()".to_string(),
                                "GLboolean *" => param.ctype = "*mut GLboolean".to_string(),
                                "GLchar *" => param.ctype = "*mut GLchar".to_string(),
                                "GLcharARB *" => param.ctype = "*mut GLcharARB".to_string(),
                                "GLenum *" => param.ctype = "*mut GLenum".to_string(),
                                "GLbyte *" => param.ctype = "*mut GLbyte".to_string(),
                                "GLubyte *" => param.ctype = "*mut GLubyte".to_string(),
                                "GLshort *" => param.ctype = "*mut GLshort".to_string(),
                                "GLushort *" => param.ctype = "*mut GLushort".to_string(),
                                "GLint *" => param.ctype = "*mut GLint".to_string(),
                                "GLuint *" => param.ctype = "*mut GLuint".to_string(),
                                "GLint64 *" => param.ctype = "*mut GLint64".to_string(),
                                "GLuint64 *" => param.ctype = "*mut GLuint64".to_string(),
                                "GLint64EXT *" => param.ctype = "*mut GLint64EXT".to_string(),
                                "GLuint64EXT *" => param.ctype = "*mut GLuint64EXT".to_string(),
                                "GLfloat *" => param.ctype = "*mut GLfloat".to_string(),
                                "GLdouble *" => param.ctype = "*mut GLdouble".to_string(),
                                "GLfixed *" => param.ctype = "*mut GLfixed".to_string(),
                                "GLhalfNV *" => param.ctype = "*mut GLhalfNV".to_string(),
                                "GLsizei *" => param.ctype = "*mut GLsizei".to_string(),
                                "GLintptr *" => param.ctype = "*mut GLintptr".to_string(),
                                "GLsizeiptr *" => param.ctype = "*mut GLsizeiptr".to_string(),
                                "const void *" => param.ctype = "*const ()".to_string(),
                                "const void **" => param.ctype = "*const *const ()".to_string(),
                                "const void *const*" => param.ctype = "*const *const ()".to_string(),
                                "const GLboolean *" => param.ctype = "*const GLboolean".to_string(),
                                "const GLboolean **" => param.ctype = "*const *const GLboolean".to_string(),
                                "const GLchar *" => param.ctype = "*const GLchar".to_string(),
                                "const GLchar *const*" => param.ctype = "*const *const GLchar".to_string(),
                                "const GLchar **" => param.ctype = "*const *mut GLchar".to_string(),
                                "const GLcharARB *" => param.ctype = "*const GLcharARB".to_string(),
                                "const GLcharARB **" => param.ctype = "*const *mut GLcharARB".to_string(),
                                "const GLenum *" => param.ctype = "*const GLenum".to_string(),
                                "const GLbyte *" => param.ctype = "*const GLbyte".to_string(),
                                "const GLubyte *" => param.ctype = "*const GLubyte".to_string(),
                                "const GLshort *" => param.ctype = "*const GLshort".to_string(),
                                "const GLushort *" => param.ctype = "*const GLushort".to_string(),
                                "const GLint *" => param.ctype = "*const GLint".to_string(),
                                "const GLint* " => param.ctype = "*const GLint".to_string(),
                                "const GLuint *" => param.ctype = "*const GLuint".to_string(),
                                "const GLint64 *" => param.ctype = "*const GLint64".to_string(),
                                "const GLuint64 *" => param.ctype = "*const GLuint64".to_string(),
                                "const GLint64EXT *" => param.ctype = "*const GLint64EXT".to_string(),
                                "const GLuint64EXT *" => param.ctype = "*const GLuint64EXT".to_string(),
                                "const GLfloat *" => param.ctype = "*const GLfloat".to_string(),
                                "const GLdouble *" => param.ctype = "*const GLdouble".to_string(),
                                "const GLclampf *" => param.ctype = "*const GLclampf".to_string(),
                                "const GLfixed *" => param.ctype = "*const GLfixed".to_string(),
                                "const GLhalfNV *" => param.ctype = "*const GLhalfNV".to_string(),
                                "const GLsizei *" => param.ctype = "*const GLsizei".to_string(),
                                "const GLintptr *" => param.ctype = "*const GLintptr".to_string(),
                                "const GLsizeiptr *" => param.ctype = "*const GLsizeiptr".to_string(),
                                "const GLvdpauSurfaceNV *" => param.ctype = "*const GLvdpauSurfaceNV".to_string(),
                                "struct _cl_context *" => param.ctype = "*mut _cl_context".to_string(),
                                "struct _cl_event *" => param.ctype = "*mut _cl_event".to_string(),
                                "GLhandleARB *" => param.ctype = "*mut GLhandleARB".to_string(),
                                _ => { },
                            }
                            function.params.push(param);
                        },
                        "alias" => {
                            for attr in attributes {
                                if attr.name.local_name.as_str() == "name" {
                                    function.alias = Some(attr.value);
                                }
                            }
                            self.expect_end_element();
                        },
                        _ => depth += 1,
                    }
                },
                XmlEvent::EndElement { .. } => {
                    depth -= 1;
                    if depth < 0 {
                        return function;
                    }
                },
                _ => { },
            }
        }
        panic!("unexpected end of file");
    }

    fn parse_commands(&mut self,attrs: Vec<OwnedAttribute>) {
        while let Ok(xml) = self.parser.next() {
            match xml {
                XmlEvent::StartElement { name,attributes,.. } => {
                    match name.local_name.as_str() {
                        "command" => {
                            let function = self.parse_command(attributes);
                            self.functions.push(function);
                        },
                        _ => { },
                    }
                },
                XmlEvent::EndElement { .. } => return,
                _ => { },
            }
        }
        panic!("unexpected end of file");
    }

    fn parse_registry(&mut self) {
        if let Ok(xml) = self.parser.next() {
            if let XmlEvent::StartElement { name,.. } = xml {
                if name.local_name.as_str() != "registry" {
                    panic!("registry expected");
                }
            }
            else {
                panic!("registry expected");
            }
        }
        else {
            panic!("unexpected end of file");
        }
        while let Ok(xml) = self.parser.next() {
            match xml {
                XmlEvent::StartElement { name,attributes,.. } => {
                    match name.local_name.as_str() {
                        "comment" => self.parse_comment(),
                        "types" => self.parse_types(),
                        "enums" => self.parse_enums(attributes),
                        "feature" => self.parse_feature(attributes),
                        "extensions" => self.parse_extensions(attributes),
                        "commands" => self.parse_commands(attributes),
                        _ => { panic!("unexpected element {}",name.local_name); },
                    }
                },
                XmlEvent::EndElement { .. } => {
                    return;
                },
                _ => { },
            }
        }
        panic!("unexpected end of file");    
    }
}

fn mangle(name: &str) -> String {
    name.to_uppercase()
}

fn main() {
    let mut main = Main::new("gl.xml").expect("unable to open file");
    main.expect_start_document();
    main.parse_registry();

    println!("// types (TODO: convert from C)");
    println!("/*");
    for ty in &main.types {
        if (ty.ctype.len() != 0) && (ty.name.len() != 0) && !ty.requires_khrplatform {
            println!("typedef {} {};",ty.ctype,ty.name);
        }
    }
    println!("*/");
    
    println!("");
    println!("// bit masks");
    for en in &main.bitmasks {
        println!("pub const {}: u64 = {};",en.name,en.value);
    }

    println!("");
    println!("// other enum constants");
    for en in &main.enums {
        if en.value.len() > 10 {
            println!("pub const {}: u64 = {};",en.name,en.value);
        }
        else {
            println!("pub const {}: u32 = {};",en.name,en.value);
        }
    }

    println!("");
    println!("// function types");
    for function in &main.functions {
        print!("type {} = Option<unsafe extern \"C\" fn(",mangle(&function.name));
        let mut first_param = true;
        for param in &function.params {
            if !first_param {
                print!(",");
            }
            print!("{}: {}",param.name,param.ctype);
            first_param = false;
        }
        println!(")>;");
    }

    println!("");
    println!("// functions");
    for function in &main.functions {
        println!("pub static mut {}: {} = Option::None;",function.name,mangle(&function.name));
    }

    println!("");
    println!("// function loader");
    println!("fn gl_load_symbol<T>(name: &str) -> T {{");
    println!("    let cname = CString::new(name).unwrap().into_raw();");
    println!("    let proc_address = unsafe {{ glGetProcAddress(cname as *const u8) }};");
    println!("    unsafe {{ std::mem::transmute(proc_address) }}");
    println!("}}");
    println!("");
    println!("pub fn gl_load_symbols() {{");
    for function in &main.functions {
        println!("    {} = gl_load_symbol(\"{}\");",function.name,function.name);
    }
    println!("}}");
}
