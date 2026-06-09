from src.agent.run import run_agent

run_agent(
    train_path="data/train.csv",
    test_path="data/test.csv",
    output_path="data/predictions_local.csv"
)
