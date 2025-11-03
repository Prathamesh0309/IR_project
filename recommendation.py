class Recommender:
    def __init__(self, df, cluster_column="cluster"):
        self.df = df
        self.cluster_column = cluster_column
        # Example rule-based mapping: cluster → service/resource
        self.mapping = {
            0: "Animal Control Hotline",
            1: "Sanitation Department",
            2: "Noise Complaint Hotline",
            3: "Road/Infrastructure Repair",
            4: "Parks & Recreation"
        }

    def recommend(self, cluster_label):
        return self.mapping.get(cluster_label, "General City Services")
