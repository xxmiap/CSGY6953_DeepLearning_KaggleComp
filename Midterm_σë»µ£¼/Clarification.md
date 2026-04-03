Hey folks,

Hope you had great spring break!!

I would like to provide some clarification regd the midterms.

Model Selection: You should perform LoRA fine tune, Try to limit model size within ≤ 2B parameters (Max is 4B, don't exceed further)  - (Hint: For faster Inference you can limit max tokens, do batched inference, keep prompt short, Set model.eval() and use torch.no_grad()) - So these are some hints might be useful - But try to experiment it some other ways to overcome your setup prob.
External dataset usage: No external data is allowed, Please use the list of data provided in the kaggle(If you feel 50k examples would'nt feasible to run, then figure out a small subset of data, and try to do the data sampling, cleaning to make a run) .
Submission portals: 1. kaggle:Manually upload your csv in kaggle contest and check your score in leaderboard, 2. Gradescope: Submission page for report will be opened by tomorrow, Collab your teammate name while make submission in gradescope and make 1 submision/team, and Do submit your report with your github links attached to it(Make it Public repo with all your executed ipynb file available).
Grading Breakdown: 35 -Report, 15 - code, 50 - kaggle leaderboard ranks
Deadline: March 27th (If any of your team requires a extension, pls do reach out to professor)
Also pls dont put post your answers/output imgs or pasting the entire bugs in ED boards, As Edstem is meant for general discussion purpose(Not to discuss the answers). Feel free to attend the OHs or put a private mail/dms in Ed to any of the TAs to get your doubts clarified .

Thanks,