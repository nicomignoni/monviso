# Changelog

[Unreleased]

## TODO
- Improve testing

## Added
- Defined a consisted convention for naming indexed terms.
- Better detailed the quickstart.

## Changed
- The constraints' set is now simply described by a list of `cp.Constraints`. 
- Related to above, `VI` now takes the `cp.Variable` characterizing the constraint set as attribute, instead of its size `n`. 
- Switched from generator-based iterates to simple functions
- Moved from docstrings to markdown in `api.md` in order to take advantage of macros and making the docs more maintainable: a lot of parameters and returns are identical across methods.
- Moved from Sphinx to MkDocs, probably easier to use and maintain. 
- Updated examples. 

## Removed
- Removed `VI.solution`. The package should be considered as a collection of atomic functions. 
- Temporarily commented out `hagraal_2`. It will be added back later after checking for possible mistakes. 
