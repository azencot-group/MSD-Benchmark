# Instructions

1. Use Python 3.8.20.
2. pip install -r requirements.txt
3. python create_dmelodies_dataset.py --save-mid
4. python download_midi_ddsp_weights.py
5. python create_dmelodies_wave_dataset.py

# Factors of Variation

| Name                           | Type   | Number of Classes | Classes                                   |
|--------------------------------|--------|-------------------|-------------------------------------------|
| Instrument                     | Global | 4                 | {Violin, Trumpet, Saxophone, Flute}       |
| Tonic                          | Global | 12                | {C, C#, D, D#, E, F, F#, G, G#, A, A#, B} |
| Octave (of Tonic)              | Global | 1                 | {4}                                       |
| Mode                           | Global | 3                 | {Major, Minor, Blues}                     |
| Rhythm                         | Local  | 28                | 8 choose 6 onset locations                |
| Arpeggiation Direction Chord 1 | Local  | 2                 | {Up, Down}                                |
| Arpeggiation Direction Chord 2 | Local  | 2                 | {Up, Down}                                |

The total number of sequences is 16,128.

# Notes

- All sequences are of length ~2 seconds (1 bar of 4/4 in 120 BPM). The synthesis does not yield the exact same length for every melody, so it is suggested to pad silence to the end of sequences during preprocessing if needed so that all of them match the longest sequence.
- This is an initial dataset variant. It is possible to expand this dataset. For example, it is possible to add more octaves (conditioned on the natural range of each instrument), as well as have more than one possible time signature, more than one possible tempo, and more than one possible chord progression. It is also possible to increase the sequence length.
- In music it is more accurate to consider global-local factorization rather than static-dynamic factorization. For example, tonic is neither static (it cannot be inferred from one note alone) nor dynamic (in our case), and timbre (instrument) may be characterized by both static and dynamic (on a short scale) attributes, but both tonic (in our case) and timbre are definitely global.
  - Alternatively, a possible approach that may be investigated is multi-scale factorization using multi-scale time series modeling techniques, thus eliminating the dichotomy and allowing flexibility to achieve domain-independent disentanglement models. For example, the global nature of the tonic is such that a scale of one chord, coupled with its position in the progression (I-IV-V-I in our case), is sufficient to identify the tonic of the melody.
    - Multi-scale factorization also facilitates hierarchical disentanglement learning, which may be preferred in domains such as music. For example, chord progression is a level above chord.

# Powered By

- Ashis Pati, Siddharth Kumar Gururani, and Alexander Lerch. 2020. dMelodies: A Music Dataset for Disentanglement Learning. In Proceedings of the 21th International Society for Music Information Retrieval Conference, ISMIR 2020, Montreal, Canada, October 11-16, 2020, 125–133. Retrieved from http://archives.ismir.net/ismir2020/paper/000300.pdf
  - GitHub repository: https://github.com/ashispati/dmelodies_dataset
  - Adapted to our needs.
- Yusong Wu, Ethan Manilow, Yi Deng, Rigel Swavely, Kyle Kastner, Tim Cooijmans, Aaron C. Courville, Cheng-Zhi Anna Huang, and Jesse H. Engel. 2022. MIDI-DDSP: Detailed Control of Musical Performance via Hierarchical Modeling. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, OpenReview.net. Retrieved from https://openreview.net/forum?id=UseMOjWENv
  - GitHub repository: https://github.com/magenta/midi-ddsp
