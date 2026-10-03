// A three-line event bus, present for one reason: it keeps the view modules
// from importing each other in a circle.
//
// The wizard needs to trigger a refresh; the model list needs to tell the
// wizard and the onboarding checklist that new state arrived. Wiring that with
// imports makes models.js <-> wizard.js mutually dependent. With a bus, both
// only depend on this file.
//
// Events in use:
//   "refresh"  ask the model list to re-poll   emit(evt, silent)
//   "state"    new /api/state arrived          emit(evt, stateObject)
//   "chat"     open the Chat tab on a model    emit(evt, modelId)
//   "load-chat" load a model, then open Chat   emit(evt, modelId)
//   "profile-save" open Save as profile        emit(evt, {id, backend})
const handlers = {};

export function on(evt, fn) {
  (handlers[evt] || (handlers[evt] = [])).push(fn);
}

export function emit(evt, ...args) {
  for (const fn of handlers[evt] || []) {
    try { fn(...args); } catch (e) { console.error(`bus "${evt}"`, e); }
  }
}
