import { Sokudan, type Questions, type NoulAnswer } from '../src/index.js';

const client = new Sokudan();
const response = await client.predict({
  state: '請求が二重になっています',
  questions: {
    department: { type: 'choice', instructions: '部署は', criteria: { 請求: null, 技術: '障害' } },
    urgency: { type: 'score', instructions: '緊急度は', criteria: ['低', '高'] },
    churn: { type: 'bool', instructions: '解約したいか' },
  },
});
const label: '請求' | '技術' = response.answers.department.choice;
const score: number = response.answers.urgency.score;
const noul: NoulAnswer = response.answers.churn;
void [label, score, noul];
// @ts-expect-error Choice questions do not return a score.
response.answers.department.score;
// @ts-expect-error Unknown question keys must not be inferred as valid.
response.answers.missing;
// @ts-expect-error Only the supplied choice labels are present.
response.answers.department.probabilities.sales;
// @ts-expect-error The server rejects null state.
client.predict({ state: null, questions: {} });
// @ts-expect-error Choice requires criteria.
const bad: Questions = { x: { type: 'choice', instructions: 'missing criteria' } };
void bad;

// JSON converts numeric object keys into strings on the wire.
const numeric = await client.predict({ state: 'text', questions: {
  value: { type: 'choice', instructions: 'choose', criteria: { 1: 'one', 2: 'two' } },
} });
const numericLabel: '1' | '2' = numeric.answers.value.choice;
void numericLabel;

const dynamic: Questions = {};
const dynamicResponse = await client.predict({ state: 'text', questions: dynamic });
const answer = dynamicResponse.answers.any;
if (answer?.type === 'choice') {
  const choice: string = answer.choice;
  void choice;
}
