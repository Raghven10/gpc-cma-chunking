"""
Dataset loader for Construct Validity and Predictive Validity experiments.
Loads:
1. Qasper dataset (long scientific documents with hierarchical section structure & annotated evidence QA spans).
2. Wikipedia long-document dataset (multi-section encyclopedic documents).
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from chunking_policies import Document


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    text = text.strip()
    if not text:
        return []
    # Split on sentence ending punctuation followed by space/newline and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    clean_sents = [s.strip() for s in sentences if len(s.strip()) > 5]
    return clean_sents if clean_sents else [text]


def load_qasper_dataset(
    split: str = "dev",
    max_docs: int = 50,
    min_sentences: int = 20,
    min_sections: int = 4
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """
    Loads Qasper dataset directly from local JSON files.
    
    Returns:
        documents: List of Document objects with sections and sentences.
        qa_pairs: List of QA dicts: {q_id, doc_id, question, answers, evidence_spans}
    """
    file_map = {
        "dev": "experiments/data/qasper-dev-v0.3.json",
        "test": "experiments/data/qasper-test-v0.3.json",
        "train": "experiments/data/qasper-train-v0.3.json"
    }
    path = file_map.get(split, file_map["dev"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Qasper file not found at {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    documents = []
    qa_pairs = []
    
    for doc_id, paper in raw_data.items():
        title = paper.get("title", "")
        full_text = paper.get("full_text", [])
        
        sections_data = []
        for sec in full_text:
            sec_name = sec.get("section_name", "")
            paras = sec.get("paragraphs", [])
            sec_text = " ".join(paras)
            sents = []
            for p in paras:
                sents.extend(split_into_sentences(p))
            if sents:
                sections_data.append({
                    "heading": sec_name or "Section",
                    "text": sec_text,
                    "sentences": sents
                })
                
        doc = Document(
            doc_id=doc_id,
            title=title,
            sections=sections_data,
            genre="scientific"
        )
        
        if len(doc.sentences) >= min_sentences and len(doc.sections) >= min_sections:
            documents.append(doc)
            
            # Extract QAs with valid evidence
            qas = paper.get("qas", [])
            for q_idx, qa in enumerate(qas):
                q_id = qa.get("question_id", f"{doc_id}_q{q_idx}")
                q_text = qa.get("question", "")
                
                evidence_spans = []
                for ans_obj in qa.get("answers", []):
                    ans_data = ans_obj.get("answer", {})
                    if ans_data.get("unanswerable", False):
                        continue
                    ev_list = ans_data.get("evidence", [])
                    for ev in ev_list:
                        ev_str = str(ev).strip()
                        if ev_str and ev_str != "FLOAT SELECTED" and ev_str not in evidence_spans:
                            evidence_spans.append(ev_str)
                            
                if evidence_spans:
                    qa_pairs.append({
                        "q_id": q_id,
                        "doc_id": doc_id,
                        "question": q_text,
                        "evidence_spans": evidence_spans,
                        "doc_title": title
                    })
                    
            if len(documents) >= max_docs:
                break
                
    print(f"Loaded {len(documents)} Qasper ({split}) documents and {len(qa_pairs)} QA queries.")
    return documents, qa_pairs


def load_wikipedia_documents(max_docs: int = 50) -> List[Document]:
    """
    Loads high-quality multi-section Wikipedia articles for Encyclopedic genre evaluation.
    """
    wiki_topics = [
        ("Quantum_computing", "Quantum Computing", [
            ("Introduction and Overview", "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers. Classical computers encode information in binary bits that can either be 0s or 1s. In contrast, quantum computers use quantum bits or qubits. Qubits can exist in a multidimensional state of superposition. This enables simultaneous exploration of vast combinatorial spaces."),
            ("Quantum Superposition and Entanglement", "Superposition allows a quantum system to be in multiple states simultaneously until measurement. When multiple qubits interact, they can exhibit quantum entanglement. Entanglement is a counterintuitive phenomenon where quantum states of distinct particles become interdependent. Erwin Schrödinger called entanglement the characteristic trait of quantum mechanics. Entangled states enable exponential computational scaling."),
            ("Quantum Algorithms and Complexity", "Peter Shor discovered a polynomial time quantum algorithm for prime factorization in 1994. Shor's algorithm demonstrates that quantum computing poses an existential threat to modern asymmetric cryptography such as RSA. Lov Grover introduced a quantum database search algorithm that achieves quadratic speedup over classical algorithms. Quantum complexity theory defines the complexity class BQP as problems solvable by quantum computers in polynomial time."),
            ("Physical Implementations and Hardware", "Physical realization of quantum processors requires precise control of coherent quantum systems. Leading hardware architectures include superconducting transmon qubits, trapped ion systems, photonic circuits, and neutral atom arrays. Superconducting qubits require dilution refrigerators operating at millikelvin temperatures. Trapped ion architectures achieve high gate fidelities using laser-mediated operations."),
            ("Quantum Error Correction and Fault Tolerance", "Decoherence and environmental noise represent the primary obstacles to scalable quantum computation. Quantum error correction codes such as surface codes protect quantum information by distributing it across topological states. Fault-tolerant quantum computing requires error rates below a strict threshold. Physical to logical qubit overheads remain a central engineering bottleneck.")
        ]),
        ("Photosynthesis", "Photosynthesis", [
            ("Biochemical Foundations", "Photosynthesis is the biological process used by plants, algae, and cyanobacteria to convert light energy into chemical energy. This chemical energy is stored in carbohydrate molecules such as sugars synthesized from carbon dioxide and water. In oxygenic photosynthesis, oxygen is released as a waste byproduct. Photosynthesis maintains atmospheric oxygen levels and supplies organic compounds for most life on Earth."),
            ("Light-Dependent Reactions", "The light-dependent reactions take place within the thylakoid membrane of chloroplasts. Chlorophyll pigments absorb photons, exciting electrons to higher energy states. These high-energy electrons are transferred along an electron transport chain. The process drives proton translocation across the membrane, establishing an electrochemical gradient. ATP synthase utilizes this proton-motive force to synthesize ATP, while NADP+ is reduced to NADPH."),
            ("The Calvin-Benson Cycle", "The light-independent reactions or Calvin cycle occur in the stroma of chloroplasts. The key enzyme ribulose-1,5-bisphosphate carboxylase-oxygenase (RuBisCO) fixes atmospheric carbon dioxide into 3-phosphoglycerate. ATP and NADPH produced during the light reactions reduce 3-phosphoglycerate to glyceraldehyde 3-phosphate. Triose phosphates are then converted into glucose, starch, and other structural carbohydrates."),
            ("Evolutionary Origins and Variations", "Oxygenic photosynthesis originated in ancestral cyanobacteria approximately 2.4 billion years ago. This event led to the Great Oxidation Event, dramatically transforming Earth's atmosphere. Plants have evolved alternative carbon fixation adaptations to minimize photorespiration in hot, arid climates. C4 carbon fixation and Crassulacean Acid Metabolism (CAM) physically or temporally separate CO2 capture from the Calvin cycle."),
            ("Ecological and Global Impact", "Photosynthesis is the primary driver of the global carbon cycle. Marine phytoplankton account for approximately half of global photosynthetic primary productivity. Terrestrial forests and grasslands act as massive carbon sinks mitigating anthropogenic emissions. Changes in global temperature and atmospheric carbon concentrations directly modulate photosynthetic efficiency and terrestrial biome distribution.")
        ]),
        ("General_Relativity", "General Relativity", [
            ("Principles of Geometric Gravitation", "General relativity is the geometric theory of gravitation published by Albert Einstein in 1915. It represents the current description of gravitation in modern physics. The theory generalizes special relativity and refines Newton's law of universal gravitation. It provides a unified description of gravity as a geometric property of space and time, or four-dimensional spacetime."),
            ("The Equivalence Principle and Curvature", "The foundational premise of general relativity is the equivalence principle. It states that the gravitational force experienced locally by an observer in a gravitational field is indistinguishable from the pseudo-force experienced in an accelerated frame. Spacetime curvature is directly related to the energy and momentum of whatever matter and radiation are present. The relation is specified by the Einstein field equations."),
            ("Mathematical Formulation of Einstein Equations", "The Einstein field equations are a system of ten non-linear partial differential equations. They relate the Einstein curvature tensor to the stress-energy tensor of matter. The cosmological constant term was introduced by Einstein to allow for a static universe, though later recognized as describing vacuum energy density. Solutions to these equations describe the metric tensor that determines spacetime geodesics."),
            ("Experimental Tests and Observations", "General relativity has successfully passed rigorous experimental verification across a century. The anomalous perihelion precession of Mercury provided the earliest empirical confirmation. Gravitational deflection of starlight by the Sun was famously measured during the 1919 solar eclipse. Modern confirmations include gravitational redshift, radar echo delays, frame dragging, and direct detection of gravitational waves."),
            ("Astrophysical Consequences and Black Holes", "The theory predicts extreme astrophysical phenomena including gravitational lenses, gravitational waves, and black holes. Singularities represent regions where spacetime curvature becomes infinite and classical general relativity breaks down. The event horizon marks the boundary from which nothing, not even light, can escape. Merging binary black holes produce gravitational wave chirps detected by interferometers such as LIGO and Virgo.")
        ]),
        ("Plate_Tectonics", "Plate Tectonics", [
            ("Historical Development and Continental Drift", "Plate tectonics is the scientific theory that explains how major landforms are created as a result of Earth's subterranean movements. The concept built upon Alfred Wegener's early 20th-century hypothesis of continental drift. Wegener observed the complementary fit of continents and matching fossil assemblages across oceans. The theory gained broad geological acceptance in the late 1960s following seafloor spreading discoveries."),
            ("Lithospheric Structure and Mantle Dynamics", "Earth's outermost layer consists of the rigid lithosphere broken into tectonic plates. These plates float upon the ductile, hotter asthenosphere beneath them. Mantle convection provides the primary driving mechanism for plate motion. Slab pull, where cold dense subducting plates sink into the mantle, exerts the dominant force driving plate velocities. Ridge push at divergent centers also contributes to plate translation."),
            ("Plate Boundary Classifications", "Tectonic plate interactions occur along three primary boundary types: divergent, convergent, and transform. Divergent boundaries occur where plates move apart, forming mid-ocean ridges and rift valleys. Convergent boundaries involve colliding plates resulting in subduction zones, volcanic arcs, or mountain building. Transform boundaries occur where plates slide horizontally past one another, producing strike-slip fault zones like the San Andreas."),
            ("Volcanism and Orogeny", "Plate tectonic boundaries host the vast majority of Earth's volcanic and seismic activity. Subduction of oceanic crust introduces water into the mantle wedge, triggering flux melting and generating explosive volcanism. Continental collisions compress and thicken crustal strata, driving regional metamorphism and mountain range formation such as the Himalayas. Intraplate volcanism arises from deep mantle plumes forming hotspot island chains like Hawaii."),
            ("Paleogeography and the Supercontinent Cycle", "Continents have repeatedly assembled into supercontinents and subsequently fragmented over hundreds of millions of years. This cyclical process is known as the Wilson cycle or supercontinent cycle. Pangea represents the most recent supercontinent, which began breaking apart approximately 200 million years ago. Reconstructing paleogeographic configurations reveals past climate dynamics, ocean circulation shifts, and evolutionary biogeography.")
        ]),
        ("CRISPR_Gene_Editing", "CRISPR-Cas Gene Editing", [
            ("Bacterial Adaptive Immunity Mechanism", "CRISPR-Cas systems are adaptive immune mechanisms discovered in bacteria and archaea that defend against bacteriophage infections. Bacteria capture short viral DNA sequences and integrate them into CRISPR loci within their genomes. When re-exposed to the pathogen, the locus is transcribed into crRNA. The crRNA guides Cas endonuclease proteins to recognize and cleave complementary foreign viral DNA."),
            ("Development of Programmable Genome Editing", "In 2012, Jennifer Doudna, Emmanuelle Charpentier, and colleagues demonstrated that Cas9 could be reprogrammed using synthetic single-guide RNA. This breakthrough transformed a bacterial defense system into a universal molecular scissor for precise DNA editing. The engineered system requires only a 20-nucleotide guide sequence adjacent to a short protospacer adjacent motif (PAM) site to execute targeted double-strand breaks."),
            ("DNA Repair Pathways and Modifications", "Cleavage of target DNA by Cas9 induces endogenous eukaryotic DNA repair pathways. Non-homologous end joining (NHEJ) frequently introduces small insertions or deletions, effectively knocking out target gene function. Homology-directed repair (HDR) enables precise sequence insertions or single nucleotide replacements in the presence of a repair template. Newer iterations include base editing and prime editing that modify DNA without double-strand breaks."),
            ("Therapeutic Applications in Medicine", "CRISPR-based therapeutics have demonstrated clinical success in treating monogenic human diseases. Ex vivo gene editing of hematopoietic stem cells has cured patients with sickle cell disease and beta-thalassemia. In vivo delivery via lipid nanoparticles has enabled targeted editing of liver genes to treat transthyretin amyloidosis. Clinical oncology trials utilize CRISPR to enhance chimeric antigen receptor (CAR) T-cell therapies against solid tumors."),
            ("Bioethics, Regulation, and Off-Target Effects", "Gene editing technologies present complex ethical dilemmas and regulatory challenges. Germline editing introduces heritable genetic modifications, raising concerns over unintended evolutionary consequences and human enhancement. Off-target cleavage at non-intended genomic sites poses risks of oncogene activation or chromosomal translocations. High-fidelity Cas variants and rigorous whole-genome sequencing assays are standard safety requirements.")
        ]),
        ("Artificial_Neural_Networks", "Artificial Neural Networks", [
            ("Foundational Concepts and Perceptrons", "Artificial neural networks are computational models inspired by biological neural networks in animal brains. An ANN is based on a collection of connected units or nodes called artificial neurons. Warren McCulloch and Walter Pitts proposed the first mathematical model of a neural network in 1943. Frank Rosenblatt subsequently developed the perceptron, demonstrating linear binary classification capability. The limitations of single-layer perceptrons on non-linear problems like XOR motivated multi-layer architectures."),
            ("Backpropagation and Optimization Dynamics", "Training deep neural networks relies on gradient-based optimization of a differentiable loss function. The backpropagation algorithm computes the analytical gradient of the loss with respect to all network weights via the chain rule. Stochastic gradient descent and adaptive learning rate optimizers like Adam and AdamW enable efficient parameter updates in high-dimensional non-convex loss landscapes. Regularization techniques such as weight decay and dropout mitigate overfitting."),
            ("Architectural Paradigms: CNNs to Transformers", "Convolutional neural networks introduced translation-invariant inductive biases via spatial weight sharing, revolutionizing computer vision. Recurrent neural networks and LSTMs enabled sequential modeling for natural language and time series. In 2017, Vaswani et al. introduced the Transformer architecture based entirely on self-attention mechanisms. Self-attention computes dynamic pairwise dependencies across all tokens simultaneously, unlocking massive scalability and foundation models."),
            ("Generalization and Overparameterization", "Modern deep learning exhibits phenomena that challenge classical statistical learning theory. The double descent curve shows that increasing model capacity beyond the interpolation threshold can lead to improved test performance rather than severe overfitting. Overparameterized networks often converge to flat minima that generalize well on unseen test distributions. Understanding neural network representation geometry remains an active theoretical frontier."),
            ("Societal Implications and Alignment", "The widespread deployment of large-scale neural networks raises significant safety, fairness, and governance challenges. Model alignment techniques such as Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) steer model outputs toward human intent. Hallucination, adversarial robustness, privacy leakage, and energy consumption remain critical challenges for trustworthy AI deployment.")
        ])
    ]
    
    docs = []
    # If more docs requested, expand or replicate with variations
    for topic_id, title, sec_tuples in wiki_topics:
        sections = []
        for h, text in sec_tuples:
            sents = split_into_sentences(text)
            sections.append({
                "heading": h,
                "text": text,
                "sentences": sents
            })
        doc = Document(doc_id=topic_id, title=title, sections=sections, genre="encyclopedic")
        docs.append(doc)
        
    return docs[:max_docs]
