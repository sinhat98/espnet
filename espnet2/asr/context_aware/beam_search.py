import logging
from typing import Any, Dict, List, Tuple

import torch

from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.e2e_asr_common import end_detect


class CABeamSearch(BeamSearch):
    def search(
        self,
        running_hyps: List[Hypothesis],
        x: torch.Tensor,
        ct: torch.Tensor,
    ) -> List[Hypothesis]:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (List[Hypothesis]): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D_a)
            ct (torch.Tensor): Encoded context feature (N, D_c)

        Returns:
            List[Hypotheses]: Best sorted hypotheses

        """
        best_hyps = []
        part_ids = torch.arange(self.n_vocab, device=x.device)  # no pre-beam
        for hyp in running_hyps:
            # scoring
            weighted_scores = torch.zeros(self.n_vocab, dtype=x.dtype, device=x.device)
            scores, states = self.score_full(hyp, x, ct)
            for k in self.full_scorers:
                weighted_scores += self.weights[k] * scores[k]
            # partial scoring
            if self.do_pre_beam:
                pre_beam_scores = (
                    weighted_scores
                    if self.pre_beam_score_key == "full"
                    else scores[self.pre_beam_score_key]
                )
                part_ids = torch.topk(pre_beam_scores, self.pre_beam_size)[1]
            part_scores, part_states = self.score_partial(hyp, part_ids, x)
            for k in self.part_scorers:
                weighted_scores[part_ids] += self.weights[k] * part_scores[k]
            # add previous hyp score
            weighted_scores += hyp.score

            # update hyps
            for j, part_j in zip(*self.beam(weighted_scores, part_ids)):
                # will be (2 x beam at most)
                best_hyps.append(
                    Hypothesis(
                        score=weighted_scores[j],
                        yseq=self.append_token(hyp.yseq, j),
                        scores=self.merge_scores(
                            hyp.scores, scores, j, part_scores, part_j
                        ),
                        states=self.merge_states(states, part_states, part_j),
                    )
                )

            # sort and prune 2 x beam -> beam
            best_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)[
                : min(len(best_hyps), self.beam_size)
            ]
        return best_hyps

    def score_full(
        self,
        hyp: Hypothesis,
        x: torch.Tensor,
        ct: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature
            ct (torch.Tensor): Context features (n_context_tokens, n_state)

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        # full_scorers is the dict which has key=module's name, value=decoder's module
        for k, d in self.full_scorers.items():
            # score() is called by an instance of Decoder module
            scores[k], states[k] = d.score(hyp.yseq, hyp.states[k], x, ct)
        return scores, states

    # def score_partial(
    #     self,
    #     hyp: Hypothesis,
    #     ids: torch.Tensor,
    #     x: torch.Tensor,
    #     ct: torch.Tensor,
    # ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    #     """Score new hypothesis by `self.part_scorers`.

    #     Args:
    #         hyp (Hypothesis): Hypothesis with prefix tokens to score
    #         ids (torch.Tensor): 1D tensor of new partial tokens to score
    #         x (torch.Tensor): Corresponding input feature
    #         ct (torch.Tensor): Context feature
    #     Returns:
    #         Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
    #             score dict of `hyp` that has string keys of `self.part_scorers`
    #             and tensor score values of shape: `(len(ids),)`,
    #             and state dict that has string keys
    #             and state values of `self.part_scorers`

    #     """
    #     scores = dict()
    #     states = dict()
    #     # part_scorers is the dict which has key=module's name, value=decoder's module
    #     for k, d in self.part_scorers.items():
    #         scores[k], states[k] = d.score_partial(hyp.yseq, ids, hyp.states[k], x, ct)
    #     return scores, states

    def forward(
        self,
        x: torch.Tensor,
        ct: torch.Tensor,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D_a)
            ct (torch.Tensor): Encoded context feature (N, D_c)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
                If maxlenratio<0.0, its absolute value is interpreted
                as a constant max output length.
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        # set length bounds
        if maxlenratio == 0:
            maxlen = x.shape[0]
        elif maxlenratio < 0:
            maxlen = -1 * int(maxlenratio)
        else:
            maxlen = max(1, int(maxlenratio * x.size(0)))
        minlen = int(minlenratio * x.size(0))
        logging.info("decoder input length: " + str(x.shape[0]))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # main loop of prefix search
        running_hyps = self.init_hyp(x)
        ended_hyps = []
        for i in range(maxlen):
            logging.debug("position " + str(i))
            best = self.search(running_hyps, x, ct)
            # post process of one iteration
            running_hyps = self.post_process(i, maxlen, maxlenratio, best, ended_hyps)
            # end detection
            if maxlenratio == 0.0 and end_detect([h.asdict() for h in ended_hyps], i):
                logging.info(f"end detected at {i}")
                break
            if len(running_hyps) == 0:
                logging.info("no hypothesis. Finish decoding.")
                break
            else:
                logging.debug(f"remained hypotheses: {len(running_hyps)}")

        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return (
                []
                if minlenratio < 0.1
                else self.forward(x, ct, maxlenratio, max(0.0, minlenratio - 0.1))
            )

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logging.info(
                f"{v:6.2f} * {self.weights[k]:3} = {v * self.weights[k]:6.2f} for {k}"
            )
        logging.info(f"total log probability: {best.score:.2f}")
        logging.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        logging.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logging.info(
                "best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        return nbest_hyps
