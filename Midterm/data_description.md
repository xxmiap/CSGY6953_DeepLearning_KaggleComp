train.csv — 50,000 rows, given to participants.

train.csv — 50,000 rows, given to participants.
Column	Type	Description
id	string	Unique row identifier (UUID)
prompt	string	Natural language description of the target image (~20 words on average)
svg	string	Ground-truth SVG code


test.csv — 1,000 rows, given to participants. No SVG column — your model must generate it.
Column	Type	Description
id	string	Unique row identifier
prompt	string	Natural language description of the target image


sample_submission.csv — 1,000 rows. Shows the required submission format.
Column	Type	Description
id	string	Must match test.csv exactly, same order
svg	string	Your generated SVG code