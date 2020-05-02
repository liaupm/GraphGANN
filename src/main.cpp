#include "defines.hpp"
#include "Parser.hpp"
#include "MainClass.hpp"

int main( int argc, char** argv )
{
	Parser parser = Parser();
	parser.parseOptions( DEFAULT_PARSER_INFILE_OPTIONS );

	MainClass handler( parser );
	handler.run();
    return 0;
}