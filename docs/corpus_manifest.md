## Information On corpus_manifest.csv File ##  
The corpus_manifest.csv file is the main metadata file for the document corpus used in the InfoGuide pipeline. Its purpose is to keep all documents organized, traceable, and reproducible by storing basic information about where each document comes from. Each row represents one document in the corpus.

The file includes columns such as doc_id (a unique ID for each document), filename and file_format (used to locate and read the file), source and source_url (showing where the document was obtained), language, retrieval_date and publication_date, license (to confirm that the document can be used), and simulated_banking_role.

**Simulated Banking Role Column on corpus_manifest.csv**    
The simulated_banking_role column specifies the primary banking professional role for which a given document is assumed to be relevant and used, within a enterprise or banking scenario. It provides business context for how and why the document would be consumed inside a bank. Assigning a simulated banking role helps ground document classification, anonymization, and taxonomy decisions. Below are possible simulated banking roles:
- Compliance Officer: Responsible for ensuring adherence to laws, regulations, and supervisory requirements. This role checks whether documents and processes match existing laws and official guidelines.

- Risk Analyst / Financial Stability Analyst: Focuses on analyzing financial, operational, and systemic risks, including macroeconomic conditions, stress testing, and financial stability assessments.

- Policy Advisor / Governance Specialist: Involved in designing and evaluating institutional policies, governance frameworks, and strategic guidelines that shape organizational decision-making.

- Data Protection / Privacy Officer: Oversees personal data protection, privacy compliance, and data governance, ensuring proper handling of PII in line with data protection regulations.

- Banking Operations / Business Analyst: Concentrates on day-to-day banking operations and business processes, using documents to analyze workflows, efficiency, and business performance.

- Other: Used for documents that do not clearly fit any of the defined roles.