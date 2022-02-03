# SmithFit
Software for fitting network analyzer data

Usage:

smithfit.py fname [-h] [-steps [STEPS]] [-every [N]] [-delimiter DELIMITER] [-comments COMMENTS] [-header HEADER] [-footer FOOTER] [--save-plots] [-ofolder OFOLDER] [-oformat OFORMAT] [--plot-steps]

positional arguments:
  fname                 File Name

optional arguments:
  -h, --help            show this help message and exit
  -steps [STEPS]        Number of steps for the fit
  -every [N]            Read every N lines from file
  -delimiter DELIMITER  The string used to separate values
  -comments COMMENTS    The characters or list of characters used to indicate the start of a comment
  -header HEADER        The number of lines to skip at the beginning of the file.
  -footer FOOTER        The number of lines to skip at the end of the file.
  --save-plots          If present, the software saves the plots as image files
  -ofolder OFOLDER      Output folder
  -oformat OFORMAT      Output format (.png, .jpg, .svg, .pdf etc.)
  --plot-steps          Plot every fit iteration

For the test data provided, you should run:
'''
python3 smithfit.py data/45d_1.s1p
'''
or
'''
python3 smithfit.py data/AutoSave8.csv -delimiter ',' -header 19 -footer 1
'''

To save the plots automatically, you need to add the --save-plots flag and optionally, you can specify the folder where you want to output the plot file and the format.
