import json
import argparse
from promptsource.templates import Template

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--jinja_template",
		type=str,
		help="Template written in jinja string."
	)
	parser.add_argument(
		"--template_name",
		type=str,
		help="Name of the template."
	)
	parser.add_argument(
		"--template_reference",
		type=str,
		help="A reference for the template."
	)
	parser.add_argument(
		"--answer_choices",
		type=str,
		default=None,
		help="""Jinja expression for answer choices. Should produce
				a ||| delimited string of choices that enumerates
				the possible completions for templates that should
				be evaluated as ranked completions. If None, then
				the template is open-ended. This list is accessible
				from within Jinja as the variable `answer_choices`."""
	)
	parser.add_argument(
		"--json_example",
		type=str,
		help="Json data in example, should be parsed by json.loads()."
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=None
		help="Path where json data will be saved"
	)
	args = parser.parse_args()
 
	json_example = json.loads(args.json_example)
	template = Template(
		args.template_name, 
	 	args.jinja_template, 
	  	args.template_reference, 
	   	answer_choices=args.answer_choices
	)
	lm_io = template.apply(json_example, highlight_variables=False)
	if args.output_dir is None:
		print(f"{json.dumps(lm_io, indent=4)}")
	else:
		with open(args.output_dir, "w") as fileptr:
			fileptr.write(f"{json.dumps(lm_io, indent=4)}")

if __name__ == "__main__":
	main()