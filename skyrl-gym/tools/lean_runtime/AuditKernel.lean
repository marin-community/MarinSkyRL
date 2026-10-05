import Lean
open Lean

structure AuditState where
  env : Environment
  visiting : NameSet := {}

partial def recheck (candidate : ModuleData) (name : Name) : StateT AuditState (ExceptT String IO) Unit := do
  let state ← get
  if state.env.contains name then return
  if state.visiting.contains name then throw "cyclic untrusted declaration"
  modify fun s => {s with visiting := s.visiting.insert name}
  let some info := candidate.constants.find? (fun info => info.name == name) | throw s!"missing declaration {name}"
  let decl ← match info with
    | .thmInfo val => pure <| Declaration.thmDecl val
    | .defnInfo val => pure <| Declaration.defnDecl val
    | .axiomInfo val => pure <| Declaration.axiomDecl val
    | .opaqueInfo val => pure <| Declaration.opaqueDecl val
    | _ => throw "unsupported untrusted declaration kind"
  decl.foldExprM (fun _ expr => expr.getUsedConstants.forM (recheck candidate)) ()
  match (← get).env.addDecl {} decl with
  | .ok env => modify fun s => {s with env := env}
  | .error _ => throw s!"kernel rejected declaration {name}"

def inspect (candidate : ModuleData) (trusted : Environment) (theoremName : Name) : ExceptT String IO (Array Name) := do
  if trusted.contains theoremName then throw "requested theorem collides with a trusted declaration"
  let some (.thmInfo _) := candidate.constants.find? (fun info => info.name == theoremName) | throw "requested theorem is missing"
  let (_,checked) ← (recheck candidate theoremName).run {env := trusted}
  let some expected := trusted.find? `AuditExpected | throw "trusted expected declaration is missing"
  let witness := Declaration.thmDecl {
    name := `AuditWitness, levelParams := expected.levelParams,
    type := expected.type, value := mkConst theoremName (expected.levelParams.map Level.param)}
  match checked.env.addDecl {} witness with
  | .error _ => throw "kernel rejected task statement binding"
  | .ok _ => pure ()
  let (_,state) := ((CollectAxioms.collect theoremName).run checked.env).run {}
  return state.axioms

def main (args : List String) : IO UInt32 := do
  if args.length != 4 then throw <| IO.userError "expected candidate module, theorem, object directory, trusted module"
  initSearchPath (← findSysroot)
  if ← isInitializerExecutionEnabled then throw <| IO.userError "user initializer execution is enabled"
  searchPathRef.modify fun paths => System.FilePath.mk args[2]! :: paths
  let (candidate, _region) ← readModuleData (System.FilePath.mk args[2]! / (args[0]! ++ ".olean"))
  let trusted ← importModules #[{module := args[3]!.toName}] {}
  if !(trusted.contains `AuditExpected) then throw <| IO.userError "trusted expected declaration is missing"
  let result ← (inspect candidate trusted args[1]!.toName).run
  let (checked,axioms,reason) := match result with
    | .ok axioms => (true,axioms,"")
    | .error message => (false,#[],message)
  IO.println <| Json.compress <| Json.mkObj [
    ("theorem_name",toJson args[1]!), ("constant_kind",toJson "theorem"),
    ("axioms",toJson (axioms.map toString)), ("reason",toJson reason),
    ("kernel_checked",toJson checked), ("process_status",toJson "completed")]
  return 0
