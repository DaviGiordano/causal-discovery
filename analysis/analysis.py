def generate_markdown(datasets, algorithms):

    markdown_content = ""

    for dataset_name in datasets:
        markdown_content += f"# Dataset: {dataset_name}\n\n"
        markdown_content += f"## True graph:\n\n ![](./../results/{dataset_name}/{algorithms[0]}/true_graph.png)\n\n"

        for algo in algorithms:
            markdown_content += f"## Algorithm: {algo}\n"
            markdown_content += (
                f"![](./../results/{dataset_name}/{algo}/graph_comparison.png)\n\n"
            )

        with open(f"./{dataset_name}.md", "w") as file:
            file.write(markdown_content)

        print(f"Markdown file generated: {dataset_name}.md")
        markdown_content = ""


algorithms = [
    "pc_default",
    "fci_default",
    "ges_default",
    "es_default",
    "grasp_default",
    "icalingam_default",
    "directlingam_default",
    "notears_default",
    "grandag_default",
    "dag_gnn_default",
    "corl_default",
]

datasets = [
    "csuite_cat_chain",
    "csuite_large_backdoor",
    "csuite_mixed_confounding",
    "csuite_symprod_simpson",
    "csuite_cat_collider",
    "csuite_large_backdoor_binary_t",
    "csuite_mixed_simpson",
    "csuite_weak_arrows",
    "csuite_cat_to_cts",
    "csuite_linexp",
    "csuite_nonlingauss",
    "csuite_weak_arrows_binary_t",
    "csuite_cts_to_cat",
    "csuite_lingauss",
    "csuite_nonlin_simpson",
]

markdown_output = generate_markdown(datasets, algorithms)
