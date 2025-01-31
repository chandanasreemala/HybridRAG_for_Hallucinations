import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_halueval = pd.read_excel('generated_output_halueval_annotated.xlsx')
df_drop = pd.read_excel('generated_output_drop_annotated.xlsx')
df_covid = pd.read_excel('generated_output_covid_annotated.xlsx')
df_finance = pd.read_excel('generated_output_financebench_annotated.xlsx')
df_ragtruth = pd.read_excel('generated_output_ragtruth_annotated.xlsx')
df_pubmed = pd.read_excel('generated_output_pubmed_annotated.xlsx')

df_combined = pd.concat([df_halueval, df_drop, df_ragtruth, df_covid, df_pubmed, df_finance],ignore_index=True)

# Metrics Comparision on Overall Dataset
def calculate_rag_metrics(scores):
    """
    Calculate metrics for RAG evaluation based on manual annotations
    scores: list/array of scores where:
        0 = wrong predictions (hallucinations)
        1 = correct predictions
        2 = insufficient context
    """
    total_samples = len(scores)

    # Calculate basic counts
    wrong_preds = sum(1 for score in scores if score == 0)
    correct_preds = sum(1 for score in scores if score == 1)
    insufficient_context = sum(1 for score in scores if score == 2)

    metrics = {
        # Percentage of correct predictions out of all samples
        'accuracy': (correct_preds / total_samples) * 100,

        # Percentage of wrong predictions (hallucination rate)
        'hallucination_rate': (wrong_preds / total_samples) * 100,

        # Percentage of cases with insufficient context
        'rejection_rate': (insufficient_context / total_samples) * 100,

        # Accuracy excluding insufficient context cases
        'adjusted_accuracy': (correct_preds / (correct_preds + wrong_preds)) * 100 if (correct_preds + wrong_preds) > 0 else 0
    }

    return metrics

def compare_retrievers(combined_df):
    """
    Compare metrics across different retrievers for the combined dataset.
    """
    results = {
        'Sparse': calculate_rag_metrics(combined_df['bm25_score']),
        'Dense': calculate_rag_metrics(combined_df['semantic_score']),
        'Hybrid': calculate_rag_metrics(combined_df['hybrid_score'])
    }
    return results

def plot_results(results):
    """
    Create a bar plot with a curve and percentage annotations for the metrics.
    """
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results).round(2)

    # Create a bar plot
    metrics = ['accuracy', 'hallucination_rate', 'rejection_rate', 'adjusted_accuracy']
    pastel_colors = ['#f6cf71', '#f89c74', '#87c55f', '#9eb9f3']

    fig, ax = plt.subplots(figsize=(14, 10))
    x = np.arange(len(results))  # Number of retrievers
    width = 0.2  # Width of the bars

    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        values = df.loc[metric]
        bars = ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), color=pastel_colors[i])

        # Annotate percentage values on the bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),  # Offset text by 5 points above the bar
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=15)

    # Set labels, title, and legend
    ax.set_title('RAG Metrics Comparison on all Datasets', fontsize=25, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=22)
    ax.set_xlabel('Retriever Type', fontsize=22)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results.keys(), fontsize=17, color='black')
    ax.legend(fontsize=17, loc='upper left')

    # Customize tick parameters
    ax.tick_params(axis='x', labelsize=16, labelcolor='black')
    ax.tick_params(axis='y', labelsize=16, labelcolor='black')

    # Save the plot
    plt.tight_layout()
    plt.savefig('overall_rag_metrics.png')
    plt.show()


# Calculate and display results
results = compare_retrievers(df_combined)

# Print detailed results
print("\nOverall Detailed Results:")
print("-" * 50)
for retriever, metrics in results.items():
    print(f"\n{retriever} Retriever:")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.2f}%")

# Plot the results
plot_results(results)


# Code for calculating metrics on ONLY Hallucinated samples on OVERALL dataset
def evaluate_hallucinated_samples(df):
    """
    Evaluate model performance on hallucinated samples (fail cases).
    """
    # Filter only the fail cases (hallucinated samples)
    fail_samples = df[df['label'] == 'FAIL']
    total_fail_samples = len(fail_samples)

    # Define retrievers and their score columns
    retrievers = {
        'Keyword search': 'bm25_score',
        'Semantic search': 'semantic_score',
        'Hybrid search': 'hybrid_score'
    }

    results = {}
    for retriever_name, score_column in retrievers.items():
        # Count different predictions
        correct_preds = sum(fail_samples[score_column] == 1)
        wrong_preds = sum(fail_samples[score_column] == 0)
        insufficient_context = sum(fail_samples[score_column] == 2)

        results[retriever_name] = {
            'total_fail_samples': total_fail_samples,
            'correct_predictions': correct_preds,
            'wrong_predictions': wrong_preds,
            'insufficient_context': insufficient_context,
            'accuracy_on_fails': (correct_preds / total_fail_samples) * 100,
            'hallucination_rate': (wrong_preds / total_fail_samples) * 100,
            'rejection_rate': (insufficient_context / total_fail_samples) * 100
        }

    return results

def plot_results(results):
    """
    Visualize the results using a bar plot and show percentages on the bars.
    """
    # Prepare data for plotting
    retrievers = list(results.keys())
    metrics = ['accuracy_on_fails', 'hallucination_rate', 'rejection_rate']
    metric_labels = ['Accuracy on Fails (%)', 'Hallucination Rate (%)', 'Rejection Rate (%)']
    colors = ['#87CEEB', '#FF7F50', '#90EE90']  # Colors for the bars

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot bars for each metric
    bar_width = 0.25
    x = range(len(retrievers))

    for i, metric in enumerate(metrics):
        values = [results[r][metric] for r in retrievers]
        bars = ax.bar([pos + i * bar_width for pos in x], values, bar_width, label=metric_labels[i], color=colors[i])

        # Annotate percentage values on the bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),  # Offset text by 5 points above the bar
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=15)

    # Set labels, title, and legend
    ax.set_title(' Metrics comparison on only Hallucinated Samples', fontsize=25, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=22)
    ax.set_xlabel('Retriever Type', fontsize=22)
    ax.set_xticks([pos + bar_width for pos in x])
    ax.set_xticklabels(retrievers, fontsize=17, color='black')
    ax.legend(fontsize=17, loc='upper left')

    # Customize tick parameters
    ax.tick_params(axis='x', labelsize=16, labelcolor='black')
    ax.tick_params(axis='y', labelsize=16, labelcolor='black')

    # Save the plot
    plt.tight_layout()
    plt.savefig('combined_hallucinated_samples.png')
    plt.show()

# Evaluate metrics
results = evaluate_hallucinated_samples(combined_df)

# Print detailed results
print("\nDetailed Results for Combined Dataset:")
print("-" * 70)
for retriever, metrics in results.items():
    print(f"\n{retriever} Retriever:")
    print(f"Total fail samples: {metrics['total_fail_samples']}")
    print(f"Correct predictions: {metrics['correct_predictions']} ({metrics['accuracy_on_fails']:.2f}%)")
    print(f"Wrong predictions: {metrics['wrong_predictions']} ({metrics['hallucination_rate']:.2f}%)")
    print(f"Insufficient context: {metrics['insufficient_context']} ({metrics['rejection_rate']:.2f}%)")
    print("-" * 50)

# Plot the results
plot_results(results)