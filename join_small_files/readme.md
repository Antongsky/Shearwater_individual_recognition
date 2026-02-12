This py file (join_wav.py) aims to create artificial long recording pieces from single individual manx shearwater call recordings.
Input: a directory containing different WAV files of single shearwater’s call. One individual may have several recording files. Naming of each file : id_xxxxx (e.g., 01_20250203 means ID is 01)
Function:
randomly pick [a, b] files
Rescale to sampling rate = 48KHZ
Ensure the total length (sum of audio lengths) is < 20 min
Join them together to get a longer file
Calculate individual numbers in this long file
Repeat this for n times to create n long files
Function format: join_wav(a,b,n,x)
Output: a directory with joined recording files with naming format join_ID_count (e.g., join_A022_28 means there are 28 individuals in this file)
