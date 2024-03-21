# RingAnalyzer
A data analysis toolbox for ring resonators' test data.

Two key emphases of the toolbox are efficiency and reliability, that are, to maximize the extracted information with minimal human intervention, and quantifiable index of data quality.

# How to use
An example file example_passive_ring_analysis.py is included.

First, instantiate the RingAnalyzer object, which defines the ring configurations (add/drop ports, initial FWHM guess, output files, etc.), and defines input and output data formats.  Depending on the specific ring data format, the laser sweep wavelength and power vectors are assigned to the RingAnalyzer object.

Second, define the analysis pipeline in the ring_processor() routine.  For example, import data -> deembed GC envelop -> detect peaks -> fit resonances to models.

Third, define the main() routine, including but not limited to input/output data paths, invoke parallel workers for data processing, and export the data.