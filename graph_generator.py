import pandas as pd
from neo4j import GraphDatabase
import logging  # For better feedback

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Neo4jUploader:

    def __init__(self, uri, user, password):

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logging.info("Successfully connected to Neo4j database.")
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            logging.info("Neo4j connection closed.")

    def _execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def _execute_write_transaction_fn(self, tx_function, *args, **kwargs):
        with self.driver.session() as session:
            return session.write_transaction(tx_function, *args, **kwargs)

    def _execute_write_query(self, query, parameters=None):

        def tx_function(tx):
            tx.run(query, parameters)

        self._execute_write_transaction_fn(tx_function)

    def create_constraints(self):
        constraint_queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Cause) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (sol:Treatment) REQUIRE sol.name IS UNIQUE",
        ]
        try:
            for query in constraint_queries:
                self._execute_write_query(query)
            logging.info("Constraints created or already exist.")
        except Exception as e:
            logging.error(f"Error creating constraints: {e}")
            raise

    def _prepare_dataframe(self, df):
        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.strip()
        logging.info(f"Preparing DataFrame. Columns found: {df_copy.columns.tolist()}")

        required_cols = ["post_id", "source", "node_type", "node_name", "text_orig"]
        for col in required_cols:
            if col not in df_copy.columns:
                raise ValueError(f"Missing required column in DataFrame: {col}")

        for col_name in ["comment_id", "labels", "keywords", "segment"]:
            if col_name not in df_copy.columns:
                df_copy[col_name] = None
                if col_name in ["labels", "keywords", "segment"]:
                    logging.info(f"'{col_name}' column not found, will be ignored.")
            elif col_name in ["labels", "keywords", "segment"]:
                logging.info(f"'{col_name}' column found, but will be ignored.")

        df_copy["post_id"] = df_copy["post_id"].astype(str)
        df_copy["source"] = df_copy["source"].astype(int)
        df_copy["node_type"] = df_copy["node_type"].astype(int)
        df_copy["node_name"] = df_copy["node_name"].astype(str).str.strip()
        df_copy["text_orig"] = df_copy["text_orig"].astype(str).str.strip()
        logging.info(
            "DataFrame preparation complete. 'labels', 'keywords', and 'segment' will not be used."
        )
        return df_copy

    def link_with_frequency_tx(
        self,
        tx,
        *,
        parent_label: str,
        child_label: str,
        parent_id,
        child_id,
        child_name,
        child_description=None,
        rel_type: str,
        rel_props=None,
    ):
        child_name_param = child_name if child_name is not None else ""
        child_description_param = (
            child_description if child_description is not None else ""
        )

        child_id_property_name = (
            "name" if child_label in ["Symptom", "Cause", "Treatment"] else "id"
        )
        parent_id_property_name = (
            "name" if parent_label in ["Symptom", "Cause", "Treatment"] else "id"
        )

        cypher = f"""
        MERGE (p:{parent_label} {{{parent_id_property_name}: $parent_id_val}})
        ON CREATE SET p.created_at = timestamp(), p.frequency = CASE WHEN "{parent_label}" = "Post" THEN 1 ELSE coalesce(p.frequency, 0) END

        MERGE (c:{child_label} {{{child_id_property_name}: $child_id_val}})
        ON CREATE SET
            c.name        = $child_name_val, 
            c.original_text = $child_description_val,
            c.frequency   = 0, 
            c.created_at  = timestamp()
        ON MATCH SET 
            c.name        = $child_name_val, 
            c.original_text = $child_description_val,
            c.updated_at  = timestamp()

        MERGE (p)-[r:{rel_type}]->(c)
        ON CREATE SET
            r.frequency  = 1,
            c.frequency  = coalesce(c.frequency, 0) + 1, 
            r.created_at = timestamp(),
            r += $rel_props_val 
        ON MATCH SET
            r.frequency  = r.frequency + 1, 
            r.updated_at = timestamp(),
            r += $rel_props_val 
        """
        tx.run(
            cypher,
            parent_id_val=parent_id,
            child_id_val=child_id,
            child_name_val=child_name_param,
            child_description_val=child_description_param,
            rel_props_val=rel_props if rel_props is not None else {},
        )

    def populate_graph_from_dataframe(self, df):
        if df.empty:
            logging.warning("Input DataFrame is empty. Skipping graph population.")
            return

        prepared_df = self._prepare_dataframe(df)
        logging.info(
            f"Starting to populate graph with {len(prepared_df)} rows using pre-grouping..."
        )

        unique_post_ids = prepared_df["post_id"].unique()
        for post_id_val in unique_post_ids:
            self._execute_write_query(
                "MERGE (p:Post {id: $pid}) ON CREATE SET p.created_at = timestamp(), p.frequency = 1",
                parameters={"pid": post_id_val},
            )
        logging.info(
            f"Ensured all {len(unique_post_ids)} Post nodes exist with initial frequency."
        )

        post_data_map = {}
        for post_id, group in prepared_df.groupby("post_id"):
            post_data_map[post_id] = {"symptoms": {}, "causes": {}, "treatments": {}}
            for _, row in group.iterrows():
                node_name = str(row["node_name"]).strip()
                if not node_name:
                    continue

                details = {
                    "text_orig": str(row.get("text_orig", "")),
                    "comment_id": (
                        str(row.get("comment_id"))
                        if pd.notna(row.get("comment_id"))
                        else None
                    ),
                    "source": int(row["source"]),
                }
                if row["node_type"] == 0:  # Symptom
                    post_data_map[post_id]["symptoms"][node_name] = details
                elif row["node_type"] == 1:  # Cause
                    post_data_map[post_id]["causes"][node_name] = details
                elif row["node_type"] == 2:  # Treatment
                    post_data_map[post_id]["treatments"][node_name] = details

        logging.info(f"Pre-grouped data for {len(post_data_map)} posts.")

        processed_items = 0
        total_items_to_link = sum(
            len(d["symptoms"]) + len(d["causes"]) + len(d["treatments"])
            for d in post_data_map.values()
        )

        for post_id, data in post_data_map.items():
            # Link Post -> Symptom
            for s_name, s_details in data["symptoms"].items():
                rel_props = {"context": s_details["text_orig"]}
                if (
                    s_details["source"] == 2 and s_details["comment_id"]
                ):  # Symptom identified from a comment
                    rel_props["comment_id"] = s_details["comment_id"]

                self._execute_write_transaction_fn(
                    self.link_with_frequency_tx,
                    parent_label="Post",
                    child_label="Symptom",
                    parent_id=post_id,
                    child_id=s_name,
                    child_name=s_name,
                    child_description=s_details["text_orig"],
                    rel_type="REPORTS_SYMPTOM",
                    rel_props=rel_props,
                )
                processed_items += 1

            # Link Symptom -> Cause
            for c_name, c_details in data["causes"].items():
                for s_name in data[
                    "symptoms"
                ]:  # Link all symptoms of THIS post to THIS cause
                    rel_props = {}
                    if c_details["comment_id"]:  # Cause identified from a comment
                        rel_props["comment_id"] = c_details["comment_id"]
                        rel_props["replySource"] = c_details[
                            "comment_id"
                        ]  # For schema alignment

                    self._execute_write_transaction_fn(
                        self.link_with_frequency_tx,
                        parent_label="Symptom",
                        child_label="Cause",
                        parent_id=s_name,
                        child_id=c_name,
                        child_name=c_name,
                        child_description=c_details["text_orig"],
                        rel_type="IS_RESULT_OF",
                        rel_props=rel_props,
                    )
                processed_items += 1

            # Link Cause -> Treatment
            for t_name, t_details in data["treatments"].items():
                for c_name in data[
                    "causes"
                ]:  # Link all causes of THIS post to THIS treatment
                    rel_props = {}
                    if t_details["comment_id"]:  # Treatment identified from a comment
                        rel_props["comment_id"] = t_details["comment_id"]
                        rel_props["replySource"] = t_details[
                            "comment_id"
                        ]  # For schema alignment

                    self._execute_write_transaction_fn(
                        self.link_with_frequency_tx,
                        parent_label="Cause",
                        child_label="Treatment",
                        parent_id=c_name,
                        child_id=t_name,
                        child_name=t_name,
                        child_description=t_details["text_orig"],
                        rel_type="IS_ADDRESSED_BY",
                        rel_props=rel_props,
                    )
                processed_items += 1

            if (
                total_items_to_link > 0
                and processed_items > 0
                and (
                    processed_items % 100 == 0 or processed_items == total_items_to_link
                )
            ):
                logging.info(
                    f"Processed linking for approx {processed_items}/{total_items_to_link} S/C/T items..."
                )

        logging.info(
            "Finished populating graph with primary nodes and relationships using pre-grouping."
        )

    def create_symptom_cooccurrence_relationships(self):
        query = """
        MATCH (s1:Symptom)<-[:REPORTS_SYMPTOM]-(p:Post)-[:REPORTS_SYMPTOM]->(s2:Symptom)
        WHERE s1.name < s2.name  
        MERGE (s1)-[r:SYMPTOM_COOCCURS]-(s2)
        ON CREATE SET r.frequency = 1
        ON MATCH SET r.frequency = r.frequency + 1
        """
        try:
            self._execute_write_query(query)
            logging.info("Created/updated SYMPTOM_COOCCURS relationships.")
        except Exception as e:
            logging.error(f"Error creating SYMPTOM_COOCCURS relationships: {e}")

    def clear_database_interactive(self):
        confirmation = input(
            "DANGER: Are you sure you want to delete ALL nodes and relationships? (yes/no): "
        )
        if confirmation.lower() == "yes":
            self._execute_write_query("MATCH (n) DETACH DELETE n")
            logging.info("Database cleared successfully.")
        else:
            logging.info("Database clear operation aborted.")

    # --- Sample Query and Recommendation Functions ---
    def query_common_causes_for_symptom(self, symptom_name, limit=10):
        query = """
        MATCH (s:Symptom {name: $symptom_name})-[r:IS_RESULT_OF]->(c:Cause)
        RETURN c.name AS cause, r.frequency AS frequency_of_link, c.frequency AS cause_total_frequency
        ORDER BY r.frequency DESC, c.frequency DESC
        LIMIT $limit
        """
        logging.info(
            f"\nQuerying: Most common causes for symptom '{symptom_name}' (using IS_RESULT_OF)"
        )
        results = self._execute_query(
            query, parameters={"symptom_name": symptom_name, "limit": limit}
        )
        for record in results:
            logging.info(
                f"- Cause: {record['cause']}, Link Frequency: {record['frequency_of_link']}, Cause Freq: {record['cause_total_frequency']}"
            )
        return results

    def query_symptoms_for_cause_keyword(self, cause_keyword, limit=10):
        query = """
        MATCH (s:Symptom)-[r:IS_RESULT_OF]->(c:Cause)
        WHERE c.name CONTAINS $keyword 
        RETURN s.name AS symptom, r.frequency AS frequency_of_link, s.frequency AS symptom_total_frequency
        ORDER BY r.frequency DESC, s.frequency DESC
        LIMIT $limit
        """
        logging.info(
            f"\nQuerying: Symptoms most frequently leading to causes with name containing '{cause_keyword}'"
        )
        results = self._execute_query(
            query, parameters={"keyword": cause_keyword, "limit": limit}
        )
        for record in results:
            logging.info(
                f"- Symptom: {record['symptom']}, Link Frequency: {record['frequency_of_link']}, Symptom Freq: {record['symptom_total_frequency']}"
            )
        return results

    def query_symptom_patterns(self, limit=10):
        query = """
        MATCH (s1:Symptom)-[r:SYMPTOM_COOCCURS]-(s2:Symptom)
        WHERE s1.name < s2.name
        RETURN s1.name AS symptom1, s2.name AS symptom2, r.frequency AS co_occurrence_frequency
        ORDER BY r.frequency DESC
        LIMIT $limit
        """
        logging.info(f"\nQuerying: Most common co-occurring symptom patterns")
        results = self._execute_query(query, parameters={"limit": limit})
        for record in results:
            logging.info(
                f"- Symptom Pattern: '{record['symptom1']}' AND '{record['symptom2']}', Frequency: {record['co_occurrence_frequency']}"
            )
        return results

    def query_solutions_for_cause(self, cause_name, limit=10):

        query = """
        MATCH (c:Cause {name: $cause_name})-[r:IS_ADDRESSED_BY]->(t:Treatment)
        RETURN t.name AS treatment, r.frequency AS frequency_of_link, t.frequency AS treatment_total_frequency
        ORDER BY r.frequency DESC, t.frequency DESC
        LIMIT $limit
        """
        logging.info(
            f"\nQuerying: Treatments for cause '{cause_name}' (using IS_ADDRESSED_BY)"
        )
        results = self._execute_query(
            query, parameters={"cause_name": cause_name, "limit": limit}
        )
        for record in results:
            logging.info(
                f"- Treatment: {record['treatment']}, Link Frequency: {record['frequency_of_link']}, Treatment Freq: {record['treatment_total_frequency']}"
            )
        return results

    def _recommend_solutions_via_symptoms_path(self, symptom_names_list, limit=5):

        if not symptom_names_list:
            return {}
        query = """
        UNWIND $s_names AS target_symptom_name
        MATCH (s:Symptom {name: target_symptom_name})
        MATCH (s)-[r_sc:IS_RESULT_OF]->(c:Cause)
        MATCH (c)-[r_ct:IS_ADDRESSED_BY]->(t:Treatment)
        WITH t.name AS treatment_name,
             coalesce(s.frequency, 1) AS s_freq,
             coalesce(r_sc.frequency, 1) AS sc_rel_freq,
             coalesce(c.frequency, 1) AS c_freq,
             coalesce(r_ct.frequency, 1) AS ct_rel_freq,
             coalesce(t.frequency, 1) AS t_freq
        WITH treatment_name, 
             (toFloat(sc_rel_freq) * toFloat(ct_rel_freq) * toFloat(t_freq) / (toFloat(s_freq) * toFloat(c_freq))) AS path_score
        RETURN treatment_name, sum(path_score) AS score 
        """
        results = self._execute_query(query, parameters={"s_names": symptom_names_list})

        scores = {}
        for record in results:
            scores[record["treatment_name"]] = record["score"]
        return scores

    def _recommend_solutions_via_causes_path(self, cause_names_list, limit=5):
        if not cause_names_list:
            return {}
        query = """
        UNWIND $c_names AS target_cause_name
        MATCH (c:Cause {name: target_cause_name})
        MATCH (c)-[r_ct:IS_ADDRESSED_BY]->(t:Treatment)
        WITH t.name AS treatment_name,
             coalesce(c.frequency, 1) AS c_freq,
             coalesce(r_ct.frequency, 1) AS ct_rel_freq,
             coalesce(t.frequency, 1) AS t_freq
        WITH treatment_name, 
             (toFloat(ct_rel_freq) * toFloat(t_freq) / toFloat(c_freq)) AS path_score
        RETURN treatment_name, sum(path_score) AS score
        """
        results = self._execute_query(query, parameters={"c_names": cause_names_list})

        scores = {}
        for record in results:
            scores[record["treatment_name"]] = record["score"]
        return scores

    def recommend_solutions(
        self,
        symptom_names_list=None,
        cause_names_list=None,
        limit=5,
        direct_cause_path_weight=1.5,
    ):

        if not symptom_names_list and not cause_names_list:
            logging.warning(
                "Both symptom and cause lists are empty for recommendation."
            )
            return []

        final_treatment_scores = {}

        if symptom_names_list:
            symptom_path_scores = self._recommend_solutions_via_symptoms_path(
                symptom_names_list
            )
            for treatment, score in symptom_path_scores.items():
                final_treatment_scores[treatment] = (
                    final_treatment_scores.get(treatment, 0.0) + score
                )
            logging.info(f"Scores from symptoms path: {symptom_path_scores}")

        if cause_names_list:
            cause_path_scores = self._recommend_solutions_via_causes_path(
                cause_names_list
            )
            for treatment, score in cause_path_scores.items():
                final_treatment_scores[treatment] = final_treatment_scores.get(
                    treatment, 0.0
                ) + (score * direct_cause_path_weight)
            logging.info(
                f"Scores from direct causes path (weighted): { {k: v*direct_cause_path_weight for k,v in cause_path_scores.items()} }"
            )

        if not final_treatment_scores:
            logging.info("No treatments found for the given inputs.")
            return []

        sorted_treatments = sorted(
            final_treatment_scores.items(), key=lambda item: item[1], reverse=True
        )

        recommended_solutions = []
        logging.info(f"\nCombined & Ranked Recommended Treatments (Top {limit}):")
        for treatment_name, score in sorted_treatments[:limit]:
            logging.info(f"- Treatment: {treatment_name}, Combined Score: {score:.4f}")
            recommended_solutions.append(treatment_name)

        return recommended_solutions

    def infer_solutions_from_dataframe(self, inference_df, limit=5):

        if inference_df.empty:
            logging.warning("Inference DataFrame is empty.")
            return []

        if not {"node_type", "node_name"}.issubset(inference_df.columns):
            logging.error(
                "Inference DataFrame must contain 'node_type' and 'node_name' columns."
            )
            return []

        symptom_rows = inference_df[inference_df["node_type"] == 0]
        symptom_names = list(symptom_rows["node_name"].str.strip().unique())

        cause_rows = inference_df[inference_df["node_type"] == 1]
        cause_names = list(cause_rows["node_name"].str.strip().unique())

        if not symptom_names and not cause_names:
            logging.warning(
                "No symptoms (node_type 0) or causes (node_type 1) found in the inference DataFrame."
            )
            return []

        logging.info(
            f"Inferring solutions for symptoms: {symptom_names} and known causes: {cause_names}"
        )
        return self.recommend_solutions(
            symptom_names_list=symptom_names, cause_names_list=cause_names, limit=limit
        )
