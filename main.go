// 1. Feature engineering - process of using domain knowledge to extract features from raw data via data mining techniques.
// 2. Data Collection & Data Verificaton - at least we want to know what data did we get and what our model marked as spam.
// 3. Model creation - combine data and ideas.
// 4. Serving infrastructure - how to perform inference?

package main

import (
	"fmt"
	"time"
)

type User struct {
	Name      string
	Email     string
	CreatedAt time.Time
}

type Message struct {
	User      User
	CreatedAt time.Time
	Text      string
}

func main() {
	oneMinAgo := time.Now().Add(-1 * time.Minute)
	twoMinAgo := time.Now().Add(-2 * time.Minute)

	msg := Message{
		User:      User{Name: "John", Email: "10minutemail.com", CreatedAt: twoMinAgo},
		CreatedAt: oneMinAgo,
		Text:      "You have WON a guaranteed 1000 cash! To get your money, click the link http://russianroulette.com?n=QJKGIGHJJGCBL",
	}

	st := IsSpam(msg)

	fmt.Println(st)
}

// Used features
//  - msg.CreatedAt - time of message sending
//  - msg.User.CreatedAt - time of user creation
//  - msg.User.Email - user email
func IsSpam(msg Message) bool {
	newUser := msg.CreatedAt.Sub(msg.User.CreatedAt) <= 1*time.Minute

	blackemail := isBlacklisted(msg.User.Email)

	if newUser && blackemail {
		return true
	}

	return false
}

var blacklist = map[string]struct{}{"10minutemail.com": struct{}{}}

func isBlacklisted(email string) bool {
	_, ok := blacklist[email]
	return ok
}
