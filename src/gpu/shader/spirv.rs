use {
    sr::*,
};

const SPIRV_MAGIC_NUMBER: u32 = 0x07230203;
const SPIRV_VERSION_13: u32 = 0x00010300;
const SPIRV_GENERATOR_MAGIC_NUMBER: u32 = 0xB5009BB1;

/*
let r: Vec<u32> = vec![SPIRV_MAGIC_NUMBER,SPIRV_VERSION_15,SPIRV_GENERATOR_MAGIC_NUMBER,0];
// length+opcode (typeid) (resultid) (op1) ... (opN)
// 1. opcapability
// 2. opextension
// 3. opextinstimport
// 4. opmemorymodel
// 5. opentrypoint
// 6. all execution-mode declarations opexecutionmode/opexecutionmodeid
// 7. opstring/opsourceextension/opsource/opsourcecontinued, opname, opmembername, opmoduleprocessed
// 8. all annotated instructions
// 9. all type declarations. constant declarations, global variable declarations, opundef, opline/opnoline, opextinst
// 10. declarations: opfunction, opfunctionparameter, opfunctionend
// 11. definitions: opfunction, opfunctionparameter, block, ..., block, opfunctionend
*/

pub fn compile_module(module: ast::Module) -> Vec<u8> {
    // TODO: convert to target AST
    // TODO: optimize in target domain
    // TODO: render output
    Vec::new()
}
