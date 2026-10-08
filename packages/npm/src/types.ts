export type JsonValue = string | number | boolean | null | JsonObject | readonly JsonValue[];
export type JsonObject = { readonly [key: string]: JsonValue };
/** Prefer a string: structured state is rendered differently by the Python model. */
export type TextEntry = string | JsonObject | readonly JsonValue[];

export interface ChoiceQuestion {
  readonly type: "choice";
  readonly instructions: TextEntry;
  readonly criteria: Readonly<Record<string, TextEntry | null>>;
}

export interface ScoreQuestion {
  readonly type: "score";
  readonly instructions: TextEntry;
  /** Two to ten nonempty levels, in ascending order. */
  readonly criteria: readonly TextEntry[];
}

export interface NoulQuestion {
  readonly type: "noul" | "bool";
  readonly instructions: TextEntry;
  readonly criteria?: { readonly true?: TextEntry | null; readonly false?: TextEntry | null } | null;
}

export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;
export type Questions = Readonly<Record<string, Question>>;

export interface SystemOneRequest<Q extends Questions = Questions> {
  readonly state: TextEntry;
  readonly questions: Q;
  /** Accepted by the server, which always uses its already-loaded model. */
  readonly model?: string | null;
}

export interface ChoiceAnswer<Label extends string = string> {
  type: "choice";
  choice: Label;
  probabilities: Record<Label, number>;
  confidence: number;
}

export interface ScoreAnswer {
  type: "score";
  score: number;
  legend: Record<string, TextEntry>;
  probabilities: Record<string, number>;
  confidence: number;
}

export interface NoulAnswer {
  type: "noul";
  noul: number;
}

export type AnswerFor<Q extends Question> = Q extends ChoiceQuestion
  ? ChoiceAnswer<`${Extract<keyof Q["criteria"], string | number>}`>
  : Q extends ScoreQuestion ? ScoreAnswer : NoulAnswer;

export interface SystemOneResponse<Q extends Questions = Questions> {
  model: string;
  answers: { -readonly [K in keyof Q]: AnswerFor<Q[K]> };
  usage: { input_tokens: number; output_tokens: number };
  sokudan: {
    state_format: "text" | "object_as_key_value_lines" | "array_as_lines";
    state_tokens: number | null;
    state_truncated: boolean | null;
    backbone_passes: number | null;
    calibrated: boolean;
    calibrated_answers: string[];
    latency_ms: number;
  };
}

export interface HealthResponse {
  status: "ok" | "not_ready";
  model: string;
  model_ref: string;
  device: string | null;
  sokudan_version: string;
  calibrated: boolean;
  calibration: { temperatures: Record<string, number>; note: string };
  state_rendering: Record<string, string>;
  noul_criteria: string;
  confidence: { choice: string; score: string };
  auth: string;
  limits: {
    max_questions: number;
    max_choice_options: number;
    score_levels: [number, number];
    max_state_chars: number;
    max_concurrency: number;
    max_queue: number;
  };
}
