const papersData = [
    {
        "title": "Evaluating and optimising performance of multispecies call recognisers for ecoacoustic restoration monitoring",
        "link": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10443330/",
        "goal": "To establish a **free and open-source protocol** for developing and optimizing **multi-species call recognisers** to monitor ecosystem restoration, specifically focusing on indicators like frog species in the Murray–Darling Basin [1-3].",
        "data": "The study utilized **831 annotated 5-minute and 1-minute sound files** recorded over two years across 20 sites [4, 5]. Data processing involved extracting **100 to 200 reference calls per species**, intentionally selecting calls of varying quality and from different geographical locations to ensure **representativeness of real-world soundscapes** [4, 6].",
        "method": "The framework employed a **binary template matching algorithm** (point matching) within the R package **monitoR** [2, 7]. Performance was optimized by adjusting three 'levers': **template selection**, user-defined **amplitude cut-offs** to isolate call structures, and **score cut-offs** (similarity thresholds) which were calibrated using **Receiver Operating Characteristic (ROC)** curves [2, 3, 8, 9].",
        "results": "Recognisers performed well for most target species, achieving **ROC values over 0.8** for species such as *L. dumerilii* and *L. fletcherii* [1, 10]. However, accuracy was poor for *L. tasmaniensis* (ROC < 0.6) because of **false positives caused by weather and other frogs** [10-12]. **Insects were identified as a major source of false detections** for several species [11, 13, 14].",
        "connection": "This research is highly relevant as it explicitly identifies **crickets as a common source of acoustic overlap and noise** that must be managed during recogniser construction [12, 15]. The study provides a **robust blueprint for distinguishing target calls from background insects** by carefully selecting frequency limits and calibrating detection thresholds, a workflow that can be directly applied to an **automatic cricket recognition project** to handle geographic variation and environmental noise [6, 15-17]."
    },
    {
        "title": "NatureLM-audio: An Audio-Language Foundation Model for Bioacoustics",
        "link": "Blog: https://www.earthspecies.org/blog/introducing-naturelm-audio-an-audio-language-foundation-model-for-bioacoustics\nPaper: https://arxiv.org/abs/2411.07186 (Corrected from Link 39 typo)",
        "goal": "Develop the first large-scale audio-language foundation model tailored for animal vocalizations, enabling natural language interaction with bioacoustic data.",
        "data": "Trained on ~4 million audio-text pairs from diverse archives (Xeno-canto, iNaturalist, Watkins Marine Mammal Database) alongside human speech and music data.",
        "method": "Jointly trained an audio encoder with a large language model (LLaMA 3.1-8B), allowing the model to \"hear\" audio tokens and generate descriptive or categorical natural language text.",
        "results": "Achieved state-of-the-art results on the BEANS-Zero benchmark; demonstrated successful zero-shot species classification and the ability to describe audio content (captioning) and count individuals.",
        "connection": "Represents the current \"frontier\" of bioacoustic AI. NatureLM-audio's ability to handle unseen taxa and provide natural language descriptions makes it an ideal candidate for a high-level \"cricket search\" tool that can be used by non-experts."
    },
    {
        "title": "Artificial intelligence (BirdNET) supplements manual methods to maximize bird species richness from acoustic data sets generated from regional monitoring",
        "link": "https://cdnsciencepub.com/doi/10.1139/cjz-2023-0044",
        "goal": "Compare manual (Listening, Visual Scanning) and automated (BirdNET) processing methods to maximize species richness and minimize human effort; propose a combined Recognizer+Validation workflow.",
        "data": "Boreal Bird Monitoring Program (Yukon): 57,756 recordings (4,813 hours) from 53 SongMeter4 ARUs; subset sampling for Listening/Visual Scanning; BirdNET-Lite run on full set with a custom species list; manual validation set of top-5 confidence recordings per species (605 recordings).",
        "method": "BirdNET (CNN-based multispecies recognizer) with parameter tuning (overlap, sensitivity, confidence threshold); Recognizer with Validation (manual review of high-confidence outputs); evaluation across nine confidence thresholds using species-level Precision/Recall/F-score approximations.",
        "results": "Recognizer with Validation confirmed 76 true positive species and reduced false positives with higher thresholds; F-score peaked around 0.60.7 (F  0.87); combining Listening + Recognizer with Validation increased species detected per ecoregion by 23%63%; BirdNET was orders of magnitude faster than manual scanning but required validation to limit false positives (noted 3 s window clipping artifacts).",
        "connection": "Directly relevant: demonstrates effective use of CNN-based multispecies recognizers, need for custom species lists, threshold tuning and manual validationpractices we should adopt for automated cricket species/call recognition to balance detection accuracy and human effort."
    },
    {
        "title": "A Review of Automated Bird Sound Recognition and Analysis in the New AI Era",
        "link": "https://www.researchgate.net/publication/388211761_A_Review_of_Automated_Bird_Sound_Recognition_and_Analysis_in_the_New_AI_Era",
        "goal": "To provide a comprehensive overview of recent advancements in machine learning, particularly deep learning (DL), to improve the accuracy and efficiency of species identification in challenging acoustic environments for biodiversity conservation.",
        "data": "The review analyzes data from large-scale repositories like Xeno-Canto, BirdCLEF competitions, and Passive Acoustic Monitoring (PAM). Data processing involves noise reduction (spectral subtraction, Wiener filtering) and spectrogram generation via STFT and MFCCs.",
        "method": "Evaluates a taxonomy of techniques including handcrafted features (MFCCs, GTCCs) and Deep Learning architectures such as CNNs for spatial features, RNNs (LSTM/GRU) for temporal dependencies, and hybrid RCNNs. It also discusses advanced models like Transformers with attention mechanisms.",
        "results": "Deep learning models demonstrate superior performance in filtering environmental interference. Specific findings include 89.5% accuracy using PEE and GTCC features on Bornean bird data and 96.45% accuracy using CNNs for classifying bird states in forest fire warning systems.",
        "connection": "The methods for bioacoustic signal processing and handling background noise (wind, overlapping calls) are directly applicable to an automatic cricket recognition project. Specifically, hybrid RCNNs are highlighted for their effectiveness in combining temporal and spatial modeling, which is ideal for capturing the repetitive, rhythmic nature of cricket chirps."
    },
    {
        "title": "Fine-Tuning BirdNET for the Automatic Ecoacoustic Monitoring of Bird Species in the Italian Alpine Forests",
        "link": "https://www.mdpi.com/2078-2489/16/8/628",
        "goal": "The primary objective was to **evaluate if fine-tuning the pre-trained BirdNET model** with local, manually annotated data and data augmentation could significantly **improve avian sound recognition** in the complex soundscapes of the Italian Alpine coniferous forests [1, 2].",
        "data": "The study used **18 hours of manually tagged audio** collected via AudioMoth units in the Tovanella watershed, supplemented by Xeno-canto recordings [3, 4]. Data processing involved windowing audio into **3-second segments**, generating **log-transformed STFT spectrograms**, and applying **augmentation techniques** like pitch shifting, time stretching, gain adjustment, and background noise addition [5, 6].",
        "method": "The researchers **fine-tuned the BirdNET-Analyzer** (v2.4, featuring an EfficientNetB0-like backbone) and calibrated **species-specific confidence thresholds** [7]. This was compared against a **baseline vanilla Convolutional Neural Network (CNN)** trained from scratch using a random search for hyperparameter optimization [8].",
        "results": "The **fine-tuned BirdNET with data augmentation achieved the highest micro-average F1-score of 0.647**, nearly doubling the scores reported in similar studies and significantly outperforming the original model (0.278) and the baseline CNN (0.440) [9, 10]. While fine-tuning drastically **boosted recall**, data augmentation provided consistent **improvements in precision** [9, 11].",
        "connection": "This research provides a **blueprint for adapting bioacoustic models to specific environments**, which is directly applicable to a **cricket recognition project** [12]. It demonstrates how to **filter non-target noise** (like geophony or other animals) and highlights the importance of **local calibration** to handle habitat-specific sound propagation and overlapping signals, which would be critical for distinguishing between various cricket types in the field [4, 13, 14]."
    },
    {
        "title": "Hearing to the Unseen: AudioMoth and BirdNET as a Cheap and Easy Method for Monitoring Cryptic Bird Species",
        "link": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10459908/",
        "goal": "Evaluate the effectiveness of using low-cost AudioMoth recorders and BirdNET software for automatically correctly identifying and detecting cryptic bird species.",
        "data": "Field recordings from AudioMoth ARUs; 90.5% and 98.4% detection success for two target species in annotated recordings; evaluated impact of confidence scores on performance.",
        "method": "Passive Acoustic Monitoring (PAM) with AudioMoth hardware and BirdNET-Analyzer software; optimized confidence score thresholds per species.",
        "results": "BirdNET precision was high (92.6% for Coal Tit, 87.8% for Short-toed Treecreeper); successful detection of cryptic species vocal activity patterns (peaking after sunrise).",
        "connection": "Demonstrates a replicable, low-cost workflow (AudioMoth + open-source CNN) for monitoring difficult-to-detect species, providing a practical template for our cricket recognition system."
    },
    {
        "title": "Analysis of BirdNET Configuration and Performance Applied to the Acoustic Monitoring of a Restored Quarry",
        "link": "https://www.mdpi.com/2076-3298/13/1/31",
        "goal": "Evaluate how configuration parameters (Overlap and Sensitivity) of BirdNET impact automated species identification performance in a real-world wetland monitoring project.",
        "data": "3 years of PAM data from a restored quarry in Spain; tested combinations of Overlap (0-1.5s) and Sensitivity (0.75-1.25) using BirdNET-Analyzer v2.4.",
        "method": "Experimental analysis of BirdNET configuration settings to minimize \"species loss\" in biodiversity inventories; used Recall as a primary performance metric.",
        "results": "Default settings were often suboptimal; higher Sensitivity (1.25) significantly reduced species loss (improved Recall); Sensitivity has a stronger impact on inventory completeness than Overlap.",
        "connection": "Crucial for setting up our cricket classifier; it proves that we must systematically tune Sensitivity and Overlap to avoid missing rare calls and to maximize the completeness of our species inventory."
    },
    {
        "title": "As the World Races to Accelerate AI, Public Institutions Must Build the Architecture That Guides Its Use (McGovern Press Release)",
        "link": "https://www.mcgovern.org/2025-press-release/",
        "goal": "Announce a $75.8M commitment from the Patrick J. McGovern Foundation to build and standardize the AI architecture for social impact and public institutions.",
        "data": "N/A (Organizational strategic focus).",
        "method": "Strategic funding and standardization of AI grant evaluations, including support for bioacoustic initiatives.",
        "results": "Significant institutional backing for the development of bioacoustic foundation models and open-source science (e.g., Earth Species Project).",
        "connection": "Highlights the strong financial and institutional interest in scaling bioacoustic AI beyond academia into large-scale conservation and public citizen science projects."
    },
    {
        "title": "Earth Species Project (ESP) Annual Report 2024",
        "link": "https://toddschulte.com/wp-content/uploads/2025/07/ESP-Annual-Report-2024.pdf",
        "goal": "Summarize ESP\u2019s 2024 achievements in bioacoustics discovery, specifically the decoding of non-human communication using frontier AI.",
        "data": "Developed several landmark models/tools: NatureLM-audio (audio language model), BirdAVES (bird audio encoder), BirdSET (multi-task benchmark), and BEBE (behavioral benchmark).",
        "method": "Development of large-scale foundation models and self-supervised learning frameworks; includes a novel \"Biodenoising\" method for cleaning bioacoustic audio without clean targets.",
        "results": "NatureLM-audio achieved state-of-the-art results in zero-shot tasks across diverse taxa, including insects; BirdAVES showed a 20% performance improvement over previous models.",
        "connection": "Provides essential high-level context on the most advanced bioacoustic AI tools available. The \"Biodenoising\" and \"BirdAVES\" frameworks are directly applicable to improving the accuracy of cricket call recognition in noisy field settings."
    },
    {
        "title": "InsectSet459: an open dataset of insect sounds for bioacoustic machine learning",
        "link": "https://arxiv.org/abs/2503.15074",
        "goal": "Present the first large-scale open dataset specifically designed for insect sound recognition to support development of novel deep-learning methods for bug bioacoustics.",
        "data": "26,399 audio files; 459 species of Orthoptera (crickets, grasshoppers) and Cicadidae (cicadas); uses varying sample rates to capture broad frequency ranges.",
        "method": "Benchmarked with two state-of-the-art deep learning classifiers to establish baseline insect recognition performance.",
        "results": "Demonstrated good initial performance but highlighted significant room for improvement in handling variable frequencies and sample rates in insect recordings.",
        "connection": "Extremely relevant; this dataset provides the primary training and evaluation data for cricket species recognition. The challenges identified in handling variable frequencies are central to building a robust recognition system for our project."
    },
    {
        "title": "Designing a BirdNET classifier for high wind detection in passive acoustic recordings to support wildlife monitoring",
        "link": "https://pubs.aip.org/asa/jasa/article/157/6/4502/3350580/Designing-a-BirdNET-classifier-for-high-wind",
        "goal": "Develop and evaluate custom BirdNET models to detect wind (low/medium/high) as unwanted sounds in Ad\u00e9lie penguin colony recordings and present a workflow for model training, evaluation, and minimum confidence threshold determination.",
        "data": "13,932 5\u2011min recordings (24 kHz, 16-bit, 6 dB gain) from seven Song Meter Minis at five Ad\u00e9lie penguin colonies; training: 100 3\u2011s clips per class (low, medium, high wind) + 100 background (penguin vocalizations); test: 78 5\u2011min files annotated into 3\u2011s segments (7800 segments); manual annotations in Raven Pro; BirdNET batch analysis (3\u2011s bins).",
        "method": "Custom BirdNET models trained via BirdNET Analyzer GUI (v1.2.0, model v2.4): four models (default/autotune \u00d7 with/without low wind class); evaluation using AUPRC/AUROC, precision, recall, F\u2011score, accuracy; binomial logistic regression to compute per\u2011class minimum confidence thresholds achieving 90% TP probability.",
        "results": "Best model (Model 1, default with low/medium/high) achieved overall F\u2011score 0.43, accuracy 0.53, precision 0.52, recall 0.37; high wind class: precision 0.76, recall 0.94; minimum confidence thresholds for 90% TP: medium 0.56, high 0.91 (low wind unachievable); autotune models failed to converge; increasing confidence threshold raised precision but lowered recall.",
        "connection": "Directly applicable: shows BirdNET can detect unwanted environmental noise (wind) to auto\u2011clean datasets; recommends manual test annotations, per\u2011class confidence thresholds (logistic regression), careful class definitions (low wind is ambiguous), and reporting raw TP/FP/FN/TN \u2014 all useful for pre\u2011filtering noisy cricket recordings and improving downstream species/call recognition."
    },
    {
        "title": "BirdNET: applications, performance, pitfalls and future opportunities",
        "link": "https://www.researchgate.net/publication/368849592_BirdNET_applications_performance_pitfalls_and_future_opportunities",
        "goal": "The paper aims to perform a **comprehensive literature review** to evaluate the current applications, performance metrics, and potential pitfalls of **BirdNET**—an automated bird sound recognizer—while providing recommendations for future research in passive acoustic monitoring.",
        "data": "The study analyzed **14 articles** identified through SCOPUS, Web of Science, and Google Scholar, focusing on species from **North America and Europe**. The data evaluated in these studies included both focal recordings and **soundscape recordings** collected via omnidirectional microphones.",
        "method": "BirdNET utilizes **convolutional neural network (CNN)** algorithms to identify vocalizations within small **3-second segments** of audio. The framework provides a **quantitative confidence score** (0 to 1) for identifications and allows users to adjust parameters such as the **confidence threshold**, **overlap** of prediction segments, and **sensitivity** levels.",
        "results": "The review found that average **precision** (correct identifications) typically ranged from **72% to 85%**, while the **recall rate** (detected target vocalizations) ranged from **33% to 84%**. Performance was highly variable among species; for example, precision for the California Scrub-Jay was 0.99, while the Common Raven could be as low as 0.29. Higher confidence thresholds were shown to increase precision but significantly decrease the recall rate.",
        "connection": "This research provides a blueprint for an **automatic cricket recognition project** by demonstrating how a **multi-species CNN classifier** can be evaluated and refined. It highlights the critical importance of selecting **species-specific confidence thresholds** and warns that performance may decrease in species-rich soundscapes where other vocally active taxa (like insects) are present, necessitating local validation for cricket-specific models."
    },
    {
        "title": "Listening for Frogs at Scale: How FrogID Evaluated NatureLM-Audio on Real-World Data",
        "link": "https://www.earthspecies.org/blog/listening-for-frogs-at-scale-how-frogid-evaluated-naturelm-audio-on-real-world-data",
        "goal": "Evaluate the multi-task performance of the NatureLM-audio foundation model on real-world citizen science data from Australia's FrogID project.",
        "data": "FrogID dataset (thousands of frog calls); evaluated tasks including Frog vs. Not-a-Frog, Species ID, Multiple Species detections, and Human Speech detection.",
        "method": "Zero-shot evaluation of the NatureLM-audio foundation model; used source separation to improve speech detection in field recordings.",
        "results": "Achieved near-perfect scores for filtering non-frog sounds; performed fairly well for Species ID (zero-shot); effectively identified human speech, which helps filter unsuitable submissions.",
        "connection": "Demonstrates the power of audio foundation models for filtering noise (insects vs. target) and identifying species without specific fine-tuning, providing a blueprint for a \"cricket-aware\" foundation model."
    },
    {
        "title": "Combining two user-friendly machine learning tools increases species detection from acoustic recordings",
        "link": "https://cdnsciencepub.com/doi/10.1139/cjz-2023-0154",
        "goal": "Evaluate a two-step sequential methodology combining Kaleidoscope Pro (clustering/HMM) and BirdNET (CNN) to improve detection of American toad calls in large field datasets.",
        "data": "~6,200 recordings from 50 ponds in northern Quebec; validation on a subset of 371 recordings.",
        "method": "Sequential approach: initial cluster scan with Kaleidoscope Pro followed by deep learning classification with BirdNET on missed/low-confidence recordings.",
        "results": "Combined approach achieved a detection rate of 93.3%, outperforming Kaleidoscope Pro alone (85.9%) and BirdNET alone (58.4%); significantly reduced expert verification time.",
        "connection": "Provides a highly practical hybrid workflow for detecting repetitive biophony (like cricket calls); the combination of HMM-based clustering with CNN-based classification is a robust strategy for high-throughput cricket monitoring."
    },
    {
        "title": "Automatic Detection and Unsupervised Clustering-Based Classification of Cetacean Vocal Signals",
        "link": "https://www.mdpi.com/2076-3417/15/7/3585",
        "goal": "Propose an automatic detection pipeline (EMD-based endpoint detection) and an unsupervised clustering-based classification approach to extract and categorize cetacean vocal signals from large PAM datasets while minimizing manual labeling.",
        "data": "194 WAV files (~25.3 h, ~1.54 GB) from Mobysound and Watkins Marine Mammal databases; five species (common dolphin, right whale, Risso's dolphin, pilot whale, beaked whale); preprocessing: DC removal, framing (25 ms), EMD decomposition, dual thresholds (TKEO and short-time ZCR) for endpoint detection; MFCC feature extraction; built 8 datasets (25 class combinations).",
        "method": "Endpoint detection via EMD + dual-parameter dual-threshold (TKEO & ZCR); MFCC feature extraction; unsupervised clustering algorithms (K-means, PCA-based clustering, GMM, SSC) evaluated across datasets; metrics: average clustering accuracy (ACC) and F-score.",
        "results": "Automatic detection reduced clutter by ~75% and extracted ~75.75 MB of vocal features from 1.58 GB of audio; clustering achieved an average accuracy of ~84.83% across datasets (GMM and K-means performed best; some datasets reached up to 100% accuracy; accuracy drops as class count increases).",
        "connection": "Relevant: demonstrates robust endpoint detection (EMD+TKEO+ZCR) and MFCC-based feature pipelines that substantially reduce manual labeling; unsupervised clustering (K-means/GMM) can identify species-level groups and may help bootstrap labeled datasets for cricket call classification in noisy environments."
    },
    {
        "title": "Foundation Models for Bioacoustics -- a Comparative Review",
        "link": "https://arxiv.org/abs/2508.01277",
        "goal": "Review large-scale pretrained bioacoustic foundation models and systematically investigate their transferability across multiple bioacoustic classification tasks (BEANS and BirdSet benchmarks).",
        "data": "Evaluation used BEANS and BirdSet benchmarks. Analysis covers model architecture, pretraining scheme (supervised vs self-supervised), and training paradigm. Data sources included large-scale bird song data and AudioSet.",
        "method": "Comprehensive review of foundation models (BirdMAE, BEATs_NLM, ConvNext_BS, Perch, etc.). Compares linear probing vs attentive probing strategies. Evaluates self-supervised learning for bioacoustic representation.",
        "results": "BirdMAE (self-supervised on bird song) performs best on BirdSet. BEATs_NLM (NatureLM-audio) is slightly better on BEANS. Self-supervised BEATs outperforms bird-specific models on BEANS when using attentive probing. Transformer-based models often require attentive probing for full performance.",
        "connection": "Vital for selecting pre-trained models for cricket call recognition. It suggests that self-supervised bird models (like BirdMAE) or general audio models (BEATs) can be adapted via probing/fine-tuning to new bioacoustic tasks (like crickets), and highlights the importance of matching the probing strategy (attentive vs linear) to the model architecture."
    },
    {
        "title": "NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics",
        "link": "https://arxiv.org/abs/2411.07186v2",
        "goal": "Present NatureLM-audio, the first audio-language foundation model specifically for bioacoustics, capable of animal vocalization detection, species classification (including zero-shot), and behavioral labeling.",
        "data": "Curated text-audio pairs spanning bioacoustics, speech, and music. Transfer learning from music/speech to bioacoustics. Evaluation on a new benchmark called BEANS-Zero.",
        "method": "Audio-language foundation model (LLM-based) prompted with audio and text. Open-sourced weights and code. Focuses on zero-shot classification and generalization to unseen taxa.",
        "results": "Sets a new state of the art on several bioacoustics tasks, including zero-shot classification of unseen species. Demonstrates that representations from music/speech can be successfully transferred to bioacoustics.",
        "connection": "Extremely promising for identifying cricket species without extensive labeled datasets (zero-shot capability). The model's ability to handle context and behavior labeling could also help in identifying different types of cricket calls (e.g., mating vs. territorial)."
    },
    {
        "title": "Advances in avian acoustic recognition through artificial intelligence: a systematic review of techniques and environmental applications",
        "link": "https://www.rbciamb.com.br/Publicacoes_RBCIAMB/article/view/2514/1184",
        "goal": "To systematically review the primary methods used in the automatic recognition of bird vocalizations, focusing on the evolution of techniques and their environmental applications for biodiversity conservation.",
        "data": "A systematic literature search was conducted across databases including Scopus, Web of Science, IEEE Xplore, and Google Scholar for the period 2013\u20132025. From an initial pool of 2,435 publications, 25 specific studies were selected and processed for in-depth analysis based on their relevance to machine learning and bioacoustics.",
        "method": "The study is an integrative literature review that examines the methodological transition from traditional feature extraction techniques, such as Mel-Frequency Cepstral Coefficients (MFCC), to advanced deep learning architectures, particularly Convolutional Neural Networks (CNNs).",
        "results": "The review identifies a significant shift toward CNNs, which offer superior classification accuracy and greater robustness against environmental noise compared to older methods. It concludes that while model generalization and computational costs remain challenges, AI-driven acoustic recognition is becoming a vital tool for real-time biodiversity monitoring.",
        "connection": "This paper is highly relevant as it documents the exact technological evolution (from MFCC to CNN) that is applicable to the automatic recognition of cricket types and calls. It provides a roadmap of proven acoustic recognition techniques and highlights the effectiveness of deep learning in handling complex biological sounds, which can be directly adapted for our project."
    },
    {
        "title": "The evolution of machine learning techniques in bird species identification: A Survey",
        "link": "https://journalwjaets.com/sites/default/files/fulltext_pdf/WJAETS-2025-0176.pdf",
        "goal": "To systematically review and evaluate machine learning algorithms (traditional like KNN, SVM and deep learning like FBN) for automated bird species classification based on their vocalizations.",
        "data": "Vocalization data from the Xeno-canto dataset. Processing involved MATLAB-based simulations and feature extraction using Mel-Frequency Cepstral Coefficients (MFCCs), spectral (centroid, bandwidth, rolloff), and timbre characteristics.",
        "method": "The study compared traditional machine learning models, specifically k-Nearest Neighbors (KNN) and Support Vector Machines (SVM), with advanced deep learning techniques such as Feedforward Backpropagation Networks (FBN).",
        "results": "Both KNN and SVM achieved 100% accuracy when utilizing MFCC and spectral features. The Feedforward Backpropagation Network (FBN) showed a slightly lower accuracy range of 95-98%.",
        "connection": "This paper is highly relevant as it demonstrates the effectiveness of MFCC feature extraction and ML classification (especially SVM and KNN) for animal acoustic signals. These methods can be directly adapted for the automatic recognition of cricket types and calls."
    },
    {
        "title": "Frog Sound Identification System for Frog Species Recognition",
        "link": "https://www.researchgate.net/publication/251880357_Frog_Sound_Identification_System_for_Frog_Species_Recognition",
        "goal": "The primary objective is to develop an automated frog sound identification system to recognize frog species from bioacoustic signals. This system aims to assist physiological researchers in locating frog species known to possess potentially beneficial antimicrobial substances.",
        "data": "The study utilized a database from AmphibiaWeb to evaluate the system's performance. The audio signals (bioacoustic recordings) were processed to extract distinctive acoustic features.",
        "method": "The system uses two main feature extraction techniques: Mel Frequency Cepstrum Coefficient (MFCC) and Linear Predictive Coding (LPC). The classification is performed using a k-Nearest Neighbor (K-NN) algorithm.",
        "results": "The experimental results demonstrated high accuracy: 98.1% accuracy when using the MFCC technique and 93.1% accuracy when using the LPC technique.",
        "connection": "This paper provides a strong methodological framework for bioacoustic signal recognition. The use of MFCC and K-NN is a standard and effective approach for sound classification, which can be directly adapted for the automatic recognition of cricket types and calls. The high accuracy achieved suggests these techniques are robust for distinguish between closely related species in audio recordings."
    }
];
