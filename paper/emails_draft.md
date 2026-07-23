# Email drafts (edit to taste before sending)

## To Simon Foreman — co-authorship invitation

Subject: Building on your RadioFisher chime-update branch — co-authorship on an RFI-masking forecast paper?

Hi Simon,

I'm Dylan Gormley, a grad student in Kevin Bandura's group at WVU. I've been
working on the DTV/RFI side of CHIME — we have an F-statistic detector for
ATSC pilot tones (PilotProxy) that produces per-channel masked-time
fractions, and the question that kept coming up was: what does that masking
actually cost in observing time for the BAO program?

To answer it I built a forecasting tool directly on your chime-update
branch of RadioFisher — the Appendix A / Table 2 configuration from the
Overview paper, your as-built baseline distribution, Tsys_tot support, and
the per-bin marginalization. The tool adds two optional noise hooks to
radiofisher (a frequency-dependent masked-time weight and a volume-excision
factor, both no-ops when absent) plus a wrapper package that turns a
masking table into required-integration-time answers in seconds. Before any
masking is applied it reproduces your Fig. 31 forecast (per-bin sigma_DV/DV
of 0.47–1.04% at 1 yr).

Given how directly this builds on your implementation, I'd like to invite
you to be a co-author — and more practically, I'd value your check on the
two places where I extended your setup: (1) the two band-averaging
conventions for heteroscedastic per-channel noise (inverse-variance vs
uniform-weight Fourier, which bracket the exact treatment), and (2) pricing
excised channels as bin-volume loss rather than noise. Draft PDF attached;
the code is at [repo links].

No pressure either way — if you'd rather just be acknowledged, that's
completely fine too. And if this should go through any CHIME internal
process given the configuration it uses, I'd appreciate your guidance on
that.

Thanks — and thanks for making the branch public; it saved this project.

Dylan

## To Phil Bull — courtesy note (not an authorship obligation)

Subject: RadioFisher lives on — RFI-masking noise hooks + a CHIME application

Hi Phil,

I'm a grad student at WVU (CHIME collaboration). I wanted to give you a
heads-up that RadioFisher is still earning citations: I've built an
RFI-masking cost forecast on top of it — per-channel masked-time fractions
from a DTV detector mapped into effective integration time and excised
volume, answering "how long must we integrate given RFI masking" for 21 cm
BAO surveys.

Concretely I forked from Simon Foreman's chime-update branch (the CHIME
Overview Appendix A setup) and added two optional hooks to baofisher.py: a
frequency-dependent noise weight w(nu) for time-domain masking, and a
vol_frac factor for excised bandwidth — both no-ops when absent, so
upstream behavior is untouched. Also some small scipy>=1.14 compatibility
fixes you're welcome to take upstream if useful.

Bull et al. (2015) is of course cited as the formalism throughout. Draft
attached in case you're curious; comments extremely welcome — and if you'd
want to be involved beyond that, happy to talk.

Best,
Dylan
