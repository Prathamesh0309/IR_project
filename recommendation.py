class Recommender:
    '''
    Recommend city services based on complaint clusters.'''

    def __init__(self):
        self.service_map = {
            "noise": "Noise — handled by NYPD",
            "garbage": "Sanitation — Department of Sanitation (DSNY)",
            "litter": "Sanitation — Department of Sanitation (DSNY)",
            "dirty": "Sanitation — Department of Sanitation (DSNY)",
            "parking": "Illegal Parking — handled by NYPD",
            "blocked driveway": "Illegal Parking — handled by NYPD",
            "rodent": "Rodent Control — NYC Health",
            "rat": "Rodent Control — NYC Health",
            "water": "Water Leaks — Department of Environmental Protection (DEP)",
            "hydrant": "Hydrant Issues — Department of Environmental Protection (DEP)",
        }
    #-------------------------------------------------------------#
    # Recommendation method
    #-------------------------------------------------------------#
    def recommend(self, results):
        """
        Generate a recommendation using:
        - Most common Complaint Type in this cluster
        - Most common Agency in this cluster
        - Optional LLM-generated cluster name
        """
        types = results["Complaint Type"].dropna().str.lower().tolist()

        if not types:
            return "General City Services"

        # Check which keyword appears most in the top similar complaints
        scores = {}
        for t in types:
            for key in self.service_map.keys():
                if key in t:
                    scores[key] = scores.get(key, 0) + 1

        if not scores:
            return "General City Services"

        # Pick the highest scoring service keyword
        best_key = max(scores, key=scores.get)
        return self.service_map[best_key]