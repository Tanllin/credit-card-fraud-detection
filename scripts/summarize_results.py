"""Read the checked-in holdout aggregates with Python's standard library."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(filename):
    with (ROOT / 'results' / filename).open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    model, = rows('final_test_model_metrics.csv')
    policy, = rows('final_test_policy_metrics.csv')
    rules = rows('final_test_rule_results.csv')
    flagged = [r for r in rules if r['decision'] in {'Hard Decline', 'Soft Decline'}]
    fraud = int(model['fraud_cases'])
    count = int(model['rows'])
    hits = sum(int(r['fraud_cases']) for r in flagged)
    alerts = sum(int(r['transactions']) for r in flagged)
    dollars_flagged = sum(float(r['fraud_amount']) for r in flagged)
    dollars_all = sum(float(r['fraud_amount']) for r in rules)
    if (sum(int(r['transactions']) for r in rules) != count
            or sum(int(r['fraud_cases']) for r in rules) != fraud
            or hits != int(policy['true_positives'])
            or alerts != int(policy['declined_or_verified'])):
        raise ValueError('The model, policy and rule summaries disagree.')

    print(f"Holdout transactions     {count:>12,}")
    print(f"Average precision        {float(model['PR-AUC']):>12.4f}")
    print(f"Interventions            {alerts:>12,}  ({alerts / count:.2%} of transactions)")
    print(f"Fraud cases flagged      {hits:>12,}  of {fraud:,} ({hits / fraud:.2%} recall)")
    print(f"Legitimate cases flagged {alerts - hits:>12,}")
    print(f"Precision among alerts   {hits / alerts:>12.2%}")
    print(f"Fraud dollars flagged    ${dollars_flagged:,.2f} of ${dollars_all:,.2f} ({dollars_flagged / dollars_all:.2%})")


if __name__ == '__main__':
    main()
